#!/usr/bin/env python3
"""
Player Performance Prediction Feature Engineering
Integrates multiple data sources to create comprehensive prediction features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text, create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

class PlayerPredictionFeatures:
    def __init__(self):
        self.engine = engine
        
    def get_player_availability_context(self, player_id, game_date, lookback_days=30):
        """
        Get player availability context including DNP patterns, rest days, etc.
        """
        query = text("""
            WITH player_recent_games AS (
                SELECT 
                    bs.game_id,
                    bs.game_date,
                    bs.player_id,
                    bs.player_name,
                    bs.minutes_played,
                    bs.comments,
                    bs.start_position,
                    -- Calculate days rest
                    LAG(bs.game_date) OVER (PARTITION BY bs.player_id ORDER BY bs.game_date) as prev_game_date,
                    -- DNP patterns
                    CASE WHEN bs.minutes_played = 0 AND bs.comments ILIKE '%DNP%' THEN 1 ELSE 0 END as dnp_count,
                    -- Injury/illness patterns
                    CASE WHEN bs.comments ILIKE '%injury%' OR bs.comments ILIKE '%illness%' THEN 1 ELSE 0 END as injury_illness_count
                FROM nba_raw.box_scores bs
                WHERE bs.player_id = :player_id
                AND bs.game_date < :game_date
                AND bs.game_date >= :game_date - INTERVAL ':lookback_days days'
                ORDER BY bs.game_date DESC
            )
            SELECT 
                player_id,
                player_name,
                COUNT(*) as games_played,
                COUNT(CASE WHEN minutes_played > 0 THEN 1 END) as games_played_actual,
                COUNT(CASE WHEN dnp_count = 1 THEN 1 END) as dnp_games,
                COUNT(CASE WHEN injury_illness_count = 1 THEN 1 END) as injury_illness_games,
                AVG(minutes_played) as avg_minutes,
                AVG(CASE WHEN minutes_played > 0 THEN minutes_played END) as avg_minutes_when_played,
                -- Recent availability trend
                AVG(CASE WHEN game_date >= :game_date - INTERVAL '7 days' THEN minutes_played END) as recent_avg_minutes,
                -- Days since last game
                EXTRACT(DAYS FROM (:game_date - MAX(game_date))) as days_since_last_game
            FROM player_recent_games
            GROUP BY player_id, player_name
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        return result.fetchone()
    
    def get_rolling_statistics(self, player_id, game_date, window_days=30):
        """
        Calculate rolling statistics for traditional, advanced, and usage metrics
        """
        query = text("""
            WITH player_stats AS (
                SELECT 
                    bs.game_date,
                    bs.minutes_played,
                    bs.points,
                    bs.field_goals_made,
                    bs.field_goals_attempted,
                    bs.three_pointers_made,
                    bs.three_pointers_attempted,
                    bs.free_throws_made,
                    bs.free_throws_attempted,
                    bs.rebounds,
                    bs.assists,
                    bs.steals,
                    bs.blocks,
                    bs.turnovers,
                    bs.personal_fouls,
                    bs.plus_minus,
                    -- Advanced stats
                    abs.off_rating,
                    abs.def_rating,
                    abs.net_rating,
                    abs.usg_pct,
                    abs.efg_pct,
                    abs.ts_pct,
                    abs.ast_pct,
                    abs.reb_pct,
                    -- Usage stats
                    bsu.usg_pct as usage_pct,
                    bsu.pct_pts as pct_team_pts,
                    bsu.pct_fga as pct_team_fga,
                    bsu.pct_ast as pct_team_ast,
                    bsu.pct_reb as pct_team_reb
                FROM nba_raw.box_scores bs
                LEFT JOIN nba_raw.advanced_box_scores abs ON bs.game_id = abs.game_id AND bs.player_id = abs.player_id
                LEFT JOIN nba_raw.box_score_usage bsu ON bs.game_id = bsu.game_id AND bs.player_id = bsu.player_id
                WHERE bs.player_id = :player_id
                AND bs.game_date < :game_date
                AND bs.game_date >= :game_date - INTERVAL ':window_days days'
                AND bs.minutes_played > 0  -- Only include games where player actually played
                ORDER BY bs.game_date DESC
            )
            SELECT 
                -- Rolling averages (last 5, 10, 20 games)
                AVG(CASE WHEN rn <= 5 THEN points END) as avg_points_5g,
                AVG(CASE WHEN rn <= 10 THEN points END) as avg_points_10g,
                AVG(CASE WHEN rn <= 20 THEN points END) as avg_points_20g,
                
                AVG(CASE WHEN rn <= 5 THEN minutes_played END) as avg_minutes_5g,
                AVG(CASE WHEN rn <= 10 THEN minutes_played END) as avg_minutes_10g,
                AVG(CASE WHEN rn <= 20 THEN minutes_played END) as avg_minutes_20g,
                
                -- Usage trends
                AVG(CASE WHEN rn <= 5 THEN usage_pct END) as avg_usage_5g,
                AVG(CASE WHEN rn <= 10 THEN usage_pct END) as avg_usage_10g,
                AVG(CASE WHEN rn <= 20 THEN usage_pct END) as avg_usage_20g,
                
                -- Efficiency trends
                AVG(CASE WHEN rn <= 5 THEN efg_pct END) as avg_efg_5g,
                AVG(CASE WHEN rn <= 10 THEN efg_pct END) as avg_efg_10g,
                AVG(CASE WHEN rn <= 20 THEN efg_pct END) as avg_efg_20g,
                
                -- Team contribution trends
                AVG(CASE WHEN rn <= 5 THEN pct_team_pts END) as avg_pct_team_pts_5g,
                AVG(CASE WHEN rn <= 10 THEN pct_team_pts END) as avg_pct_team_pts_10g,
                AVG(CASE WHEN rn <= 20 THEN pct_team_pts END) as avg_pct_team_pts_20g,
                
                -- Recent form indicators
                AVG(CASE WHEN rn <= 3 THEN points END) as recent_points_3g,
                AVG(CASE WHEN rn <= 3 THEN usage_pct END) as recent_usage_3g,
                AVG(CASE WHEN rn <= 3 THEN efg_pct END) as recent_efg_3g
                
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY game_date DESC) as rn
                FROM player_stats
            ) ranked_stats
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date,
            'window_days': window_days
        })
        
        return result.fetchone()
    
    def get_lineup_shot_patterns(self, team_id, player_id, game_date, lookback_days=60):
        """
        Analyze shot patterns for lineups that include this player
        """
        # This would integrate with the ShotChartLineupDetail endpoint
        # For now, we'll create a placeholder for the logic
        
        query = text("""
            WITH player_lineup_games AS (
                SELECT DISTINCT
                    bs.game_id,
                    bs.game_date,
                    bs.team_id,
                    bs.player_id,
                    bs.minutes_played,
                    -- Get other players in same game (potential lineup members)
                    STRING_AGG(DISTINCT other_bs.player_id::TEXT, '-' ORDER BY other_bs.player_id) as potential_lineup
                FROM nba_raw.box_scores bs
                JOIN nba_raw.box_scores other_bs ON bs.game_id = other_bs.game_id 
                    AND bs.team_id = other_bs.team_id
                    AND other_bs.minutes_played > 0
                WHERE bs.player_id = :player_id
                AND bs.game_date < :game_date
                AND bs.game_date >= :game_date - INTERVAL ':lookback_days days'
                AND bs.minutes_played > 0
                GROUP BY bs.game_id, bs.game_date, bs.team_id, bs.player_id, bs.minutes_played
            )
            SELECT 
                player_id,
                COUNT(DISTINCT game_id) as games_with_lineups,
                AVG(minutes_played) as avg_minutes_in_lineups,
                -- Most common lineup patterns (simplified)
                COUNT(DISTINCT potential_lineup) as unique_lineup_combinations
            FROM player_lineup_games
            GROUP BY player_id
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        return result.fetchone()
    
    def get_opponent_context(self, player_id, opponent_team_id, game_date, lookback_days=30):
        """
        Get player's historical performance against this opponent
        """
        query = text("""
            SELECT 
                bs.player_id,
                COUNT(*) as games_vs_opponent,
                AVG(bs.points) as avg_points_vs_opponent,
                AVG(bs.minutes_played) as avg_minutes_vs_opponent,
                AVG(bs.field_goals_made) as avg_fgm_vs_opponent,
                AVG(bs.field_goals_attempted) as avg_fga_vs_opponent,
                AVG(bs.three_pointers_made) as avg_3pm_vs_opponent,
                AVG(bs.three_pointers_attempted) as avg_3pa_vs_opponent,
                AVG(bs.rebounds) as avg_rebounds_vs_opponent,
                AVG(bs.assists) as avg_assists_vs_opponent,
                -- Recent performance vs opponent
                AVG(CASE WHEN bs.game_date >= :game_date - INTERVAL '7 days' THEN bs.points END) as recent_points_vs_opponent,
                AVG(CASE WHEN bs.game_date >= :game_date - INTERVAL '7 days' THEN bs.minutes_played END) as recent_minutes_vs_opponent
            FROM nba_raw.box_scores bs
            JOIN nba_raw.games_raw gr ON bs.game_id = gr.game_id
            WHERE bs.player_id = :player_id
            AND gr.team_id = :opponent_team_id
            AND bs.game_date < :game_date
            AND bs.game_date >= :game_date - INTERVAL ':lookback_days days'
            AND bs.minutes_played > 0
            GROUP BY bs.player_id
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'opponent_team_id': opponent_team_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        return result.fetchone()
    
    def get_season_progression_features(self, player_id, game_date):
        """
        Get season progression features like game number, rest patterns, etc.
        """
        query = text("""
            WITH season_games AS (
                SELECT 
                    bs.game_id,
                    bs.game_date,
                    bs.player_id,
                    bs.minutes_played,
                    ROW_NUMBER() OVER (PARTITION BY bs.player_id, EXTRACT(YEAR FROM bs.game_date) ORDER BY bs.game_date) as game_number_in_season,
                    LAG(bs.game_date) OVER (PARTITION BY bs.player_id ORDER BY bs.game_date) as prev_game_date
                FROM nba_raw.box_scores bs
                WHERE bs.player_id = :player_id
                AND bs.game_date < :game_date
                ORDER BY bs.game_date DESC
            )
            SELECT 
                player_id,
                COUNT(*) as total_games_this_season,
                AVG(minutes_played) as season_avg_minutes,
                -- Season progression
                AVG(CASE WHEN game_number_in_season <= 10 THEN minutes_played END) as early_season_avg_minutes,
                AVG(CASE WHEN game_number_in_season > 10 THEN minutes_played END) as recent_season_avg_minutes,
                -- Rest patterns
                AVG(EXTRACT(DAYS FROM (game_date - prev_game_date))) as avg_days_between_games,
                -- Recent rest
                EXTRACT(DAYS FROM (:game_date - MAX(game_date))) as days_since_last_game
            FROM season_games
            GROUP BY player_id
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date
        })
        
        return result.fetchone()
    
    def build_prediction_features(self, player_id, team_id, opponent_team_id, game_date):
        """
        Build comprehensive prediction features for a player in an upcoming game
        """
        features = {}
        
        # 1. Player availability context
        availability = self.get_player_availability_context(player_id, game_date)
        if availability:
            features.update({
                'games_played_last_30': availability.games_played,
                'games_played_actual_last_30': availability.games_played_actual,
                'dnp_games_last_30': availability.dnp_games,
                'injury_illness_games_last_30': availability.injury_illness_games,
                'avg_minutes_last_30': availability.avg_minutes,
                'avg_minutes_when_played_last_30': availability.avg_minutes_when_played,
                'recent_avg_minutes_7d': availability.recent_avg_minutes,
                'days_since_last_game': availability.days_since_last_game
            })
        
        # 2. Rolling statistics
        rolling_stats = self.get_rolling_statistics(player_id, game_date)
        if rolling_stats:
            features.update({
                'avg_points_5g': rolling_stats.avg_points_5g,
                'avg_points_10g': rolling_stats.avg_points_10g,
                'avg_points_20g': rolling_stats.avg_points_20g,
                'avg_minutes_5g': rolling_stats.avg_minutes_5g,
                'avg_minutes_10g': rolling_stats.avg_minutes_10g,
                'avg_minutes_20g': rolling_stats.avg_minutes_20g,
                'avg_usage_5g': rolling_stats.avg_usage_5g,
                'avg_usage_10g': rolling_stats.avg_usage_10g,
                'avg_usage_20g': rolling_stats.avg_usage_20g,
                'avg_efg_5g': rolling_stats.avg_efg_5g,
                'avg_efg_10g': rolling_stats.avg_efg_10g,
                'avg_efg_20g': rolling_stats.avg_efg_20g,
                'avg_pct_team_pts_5g': rolling_stats.avg_pct_team_pts_5g,
                'avg_pct_team_pts_10g': rolling_stats.avg_pct_team_pts_10g,
                'avg_pct_team_pts_20g': rolling_stats.avg_pct_team_pts_20g,
                'recent_points_3g': rolling_stats.recent_points_3g,
                'recent_usage_3g': rolling_stats.recent_usage_3g,
                'recent_efg_3g': rolling_stats.recent_efg_3g
            })
        
        # 3. Lineup context
        lineup_patterns = self.get_lineup_shot_patterns(team_id, player_id, game_date)
        if lineup_patterns:
            features.update({
                'games_with_lineups': lineup_patterns.games_with_lineups,
                'avg_minutes_in_lineups': lineup_patterns.avg_minutes_in_lineups,
                'unique_lineup_combinations': lineup_patterns.unique_lineup_combinations
            })
        
        # 4. Opponent context
        opponent_context = self.get_opponent_context(player_id, opponent_team_id, game_date)
        if opponent_context:
            features.update({
                'games_vs_opponent': opponent_context.games_vs_opponent,
                'avg_points_vs_opponent': opponent_context.avg_points_vs_opponent,
                'avg_minutes_vs_opponent': opponent_context.avg_minutes_vs_opponent,
                'avg_fgm_vs_opponent': opponent_context.avg_fgm_vs_opponent,
                'avg_fga_vs_opponent': opponent_context.avg_fga_vs_opponent,
                'avg_3pm_vs_opponent': opponent_context.avg_3pm_vs_opponent,
                'avg_3pa_vs_opponent': opponent_context.avg_3pa_vs_opponent,
                'avg_rebounds_vs_opponent': opponent_context.avg_rebounds_vs_opponent,
                'avg_assists_vs_opponent': opponent_context.avg_assists_vs_opponent,
                'recent_points_vs_opponent': opponent_context.recent_points_vs_opponent,
                'recent_minutes_vs_opponent': opponent_context.recent_minutes_vs_opponent
            })
        
        # 5. Season progression
        season_prog = self.get_season_progression_features(player_id, game_date)
        if season_prog:
            features.update({
                'total_games_this_season': season_prog.total_games_this_season,
                'season_avg_minutes': season_prog.season_avg_minutes,
                'early_season_avg_minutes': season_prog.early_season_avg_minutes,
                'recent_season_avg_minutes': season_prog.recent_season_avg_minutes,
                'avg_days_between_games': season_prog.avg_days_between_games,
                'days_since_last_game': season_prog.days_since_last_game
            })
        
        return features

# Example usage
if __name__ == "__main__":
    predictor = PlayerPredictionFeatures()
    
    # Example: Get features for a player in an upcoming game
    features = predictor.build_prediction_features(
        player_id=1626157,  # Example player ID
        team_id=1610612752,  # Example team ID
        opponent_team_id=1610612755,  # Example opponent
        game_date=datetime(2024, 12, 15)  # Example game date
    )
    
    print("Prediction Features:")
    for key, value in features.items():
        print(f"{key}: {value}") 