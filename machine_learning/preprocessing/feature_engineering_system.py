#!/usr/bin/env python3
"""
Comprehensive Feature Engineering System for NBA Player Prediction
Builds complete feature set for each player-game combination and saves to database
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text, create_engine
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

class FeatureEngineeringSystem:
    def __init__(self):
        self.engine = engine
        
    def build_player_game_features(self, game_date, lookback_days=30):
        """
        Build comprehensive features for all players in upcoming games
        """
        # Get all games on the specified date
        games = self._get_games_for_date(game_date)
        
        all_features = []
        
        for game in games:
            game_id = game['game_id']
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']
            
            print(f"Processing game {game_id}: {game['home_team']} vs {game['away_team']}")
            
            # Get players for both teams
            home_players = self._get_team_players(home_team_id)
            away_players = self._get_team_players(away_team_id)
            
            # Build features for home team players
            for player in home_players:
                features = self._build_player_features(
                    player['player_id'], 
                    home_team_id, 
                    away_team_id, 
                    game_date, 
                    lookback_days
                )
                if features:
                    features['game_id'] = game_id
                    features['opponent_team_id'] = away_team_id
                    all_features.append(features)
            
            # Build features for away team players
            for player in away_players:
                features = self._build_player_features(
                    player['player_id'], 
                    away_team_id, 
                    home_team_id, 
                    game_date, 
                    lookback_days
                )
                if features:
                    features['game_id'] = game_id
                    features['opponent_team_id'] = home_team_id
                    all_features.append(features)
        
        return all_features
    
    def _get_games_for_date(self, game_date):
        """Get all games scheduled for a specific date"""
        query = text("""
            SELECT 
                game_id,
                home_team_id,
                away_team_id,
                home_team,
                away_team,
                game_date
            FROM nba_analytics.games
            WHERE game_date = :game_date
            AND game_type = 'regular'
        """)
        
        result = engine.execute(query, {'game_date': game_date})
        return [dict(row) for row in result]
    
    def _get_team_players(self, team_id):
        """Get active players for a team"""
        query = text("""
            SELECT DISTINCT 
                player_id,
                player_name,
                team_id
            FROM nba_analytics.stg_box_scores
            WHERE team_id = :team_id
            AND game_date >= CURRENT_DATE - INTERVAL '30 days'
            AND minutes_played > 0
            ORDER BY player_id
        """)
        
        result = engine.execute(query, {'team_id': team_id})
        return [dict(row) for row in result]
    
    def _build_player_features(self, player_id, team_id, opponent_team_id, game_date, lookback_days):
        """
        Build comprehensive features for a specific player-game combination
        """
        features = {
            'player_id': player_id,
            'team_id': team_id,
            'game_date': game_date,
            'created_at': datetime.now()
        }
        
        # 1. Player Context Features
        player_context = self._get_player_context(player_id, game_date, lookback_days)
        if player_context:
            features.update(self._extract_player_features(player_context))
        
        # 2. Opponent Context Features
        opponent_context = self._get_opponent_context(player_id, opponent_team_id, game_date, lookback_days)
        if opponent_context:
            features.update(self._extract_opponent_features(opponent_context))
        
        # 3. Team Availability Features
        team_context = self._get_team_context(team_id, game_date, lookback_days)
        if team_context:
            features.update(self._extract_team_features(team_context))
        
        # 4. Advanced Stats Features
        advanced_features = self._get_advanced_features(player_id, game_date, lookback_days)
        if advanced_features:
            features.update(advanced_features)
        
        # 5. Usage Features
        usage_features = self._get_usage_features(player_id, game_date, lookback_days)
        if usage_features:
            features.update(usage_features)
        
        # 6. Shot Location Features
        shot_features = self._get_shot_features(player_id, game_date, lookback_days)
        if shot_features:
            features.update(shot_features)
        
        return features
    
    def _get_player_context(self, player_id, game_date, lookback_days):
        """Get player context features"""
        query = text("""
            WITH player_recent_games AS (
                SELECT 
                    bs.*,
                    abs.usg_pct, abs.efg_pct, abs.ts_pct, abs.ast_pct, abs.reb_pct,
                    abs.off_rating, abs.def_rating, abs.net_rating,
                    bsu.usg_pct as usage_pct, bsu.pct_pts, bsu.pct_fga, bsu.pct_ast, bsu.pct_reb,
                    ROW_NUMBER() OVER (ORDER BY bs.game_date DESC) as rn
                FROM nba_analytics.stg_box_scores bs
                LEFT JOIN nba_analytics.stg_advanced_box_scores abs ON bs.game_id = abs.game_id AND bs.player_id = abs.player_id
                LEFT JOIN nba_analytics.stg_box_score_usage bsu ON bs.game_id = bsu.game_id AND bs.player_id = bsu.player_id
                WHERE bs.player_id = :player_id
                AND bs.game_date < :game_date
                AND bs.game_date >= :game_date - INTERVAL ':lookback_days days'
            )
            SELECT 
                player_id,
                COUNT(*) as total_games,
                COUNT(CASE WHEN minutes_played > 0 THEN 1 END) as games_played,
                COUNT(CASE WHEN minutes_played = 0 THEN 1 END) as dnp_games,
                AVG(CASE WHEN minutes_played > 0 THEN minutes_played END) as avg_minutes,
                AVG(CASE WHEN minutes_played > 0 THEN points END) as avg_points,
                AVG(CASE WHEN minutes_played > 0 THEN field_goals_made END) as avg_fgm,
                AVG(CASE WHEN minutes_played > 0 THEN field_goals_attempted END) as avg_fga,
                AVG(CASE WHEN minutes_played > 0 THEN three_pointers_made END) as avg_3pm,
                AVG(CASE WHEN minutes_played > 0 THEN three_pointers_attempted END) as avg_3pa,
                AVG(CASE WHEN minutes_played > 0 THEN rebounds END) as avg_rebounds,
                AVG(CASE WHEN minutes_played > 0 THEN assists END) as avg_assists,
                AVG(CASE WHEN minutes_played > 0 THEN steals END) as avg_steals,
                AVG(CASE WHEN minutes_played > 0 THEN blocks END) as avg_blocks,
                AVG(CASE WHEN minutes_played > 0 THEN turnovers END) as avg_turnovers,
                AVG(CASE WHEN minutes_played > 0 THEN usg_pct END) as avg_usage,
                AVG(CASE WHEN minutes_played > 0 THEN efg_pct END) as avg_efg,
                AVG(CASE WHEN minutes_played > 0 THEN ts_pct END) as avg_ts,
                AVG(CASE WHEN minutes_played > 0 THEN ast_pct END) as avg_ast_pct,
                AVG(CASE WHEN minutes_played > 0 THEN reb_pct END) as avg_reb_pct,
                AVG(CASE WHEN minutes_played > 0 THEN off_rating END) as avg_off_rating,
                AVG(CASE WHEN minutes_played > 0 THEN def_rating END) as avg_def_rating,
                AVG(CASE WHEN minutes_played > 0 THEN net_rating END) as avg_net_rating,
                AVG(CASE WHEN minutes_played > 0 THEN usage_pct END) as avg_usage_pct,
                AVG(CASE WHEN minutes_played > 0 THEN pct_pts END) as avg_pct_team_pts,
                AVG(CASE WHEN minutes_played > 0 THEN pct_fga END) as avg_pct_team_fga,
                AVG(CASE WHEN minutes_played > 0 THEN pct_ast END) as avg_pct_team_ast,
                AVG(CASE WHEN minutes_played > 0 THEN pct_reb END) as avg_pct_team_reb,
                
                -- Recent trends (last 5 games)
                AVG(CASE WHEN rn <= 5 AND minutes_played > 0 THEN minutes_played END) as recent_avg_minutes,
                AVG(CASE WHEN rn <= 5 AND minutes_played > 0 THEN points END) as recent_avg_points,
                AVG(CASE WHEN rn <= 5 AND minutes_played > 0 THEN usage_pct END) as recent_avg_usage,
                AVG(CASE WHEN rn <= 5 AND minutes_played > 0 THEN efg_pct END) as recent_avg_efg,
                
                -- Days since last game
                EXTRACT(DAYS FROM (:game_date - MAX(game_date))) as days_since_last_game
                
            FROM player_recent_games
            GROUP BY player_id
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        return result.fetchone()
    
    def _extract_player_features(self, player_context):
        """Extract player features from context"""
        if not player_context:
            return {}
        
        return {
            'total_games': player_context.total_games or 0,
            'games_played': player_context.games_played or 0,
            'dnp_games': player_context.dnp_games or 0,
            'avg_minutes': player_context.avg_minutes or 0,
            'avg_points': player_context.avg_points or 0,
            'avg_fgm': player_context.avg_fgm or 0,
            'avg_fga': player_context.avg_fga or 0,
            'avg_3pm': player_context.avg_3pm or 0,
            'avg_3pa': player_context.avg_3pa or 0,
            'avg_rebounds': player_context.avg_rebounds or 0,
            'avg_assists': player_context.avg_assists or 0,
            'avg_steals': player_context.avg_steals or 0,
            'avg_blocks': player_context.avg_blocks or 0,
            'avg_turnovers': player_context.avg_turnovers or 0,
            'avg_usage': player_context.avg_usage or 0,
            'avg_efg': player_context.avg_efg or 0,
            'avg_ts': player_context.avg_ts or 0,
            'avg_ast_pct': player_context.avg_ast_pct or 0,
            'avg_reb_pct': player_context.avg_reb_pct or 0,
            'avg_off_rating': player_context.avg_off_rating or 0,
            'avg_def_rating': player_context.avg_def_rating or 0,
            'avg_net_rating': player_context.avg_net_rating or 0,
            'avg_usage_pct': player_context.avg_usage_pct or 0,
            'avg_pct_team_pts': player_context.avg_pct_team_pts or 0,
            'avg_pct_team_fga': player_context.avg_pct_team_fga or 0,
            'avg_pct_team_ast': player_context.avg_pct_team_ast or 0,
            'avg_pct_team_reb': player_context.avg_pct_team_reb or 0,
            'recent_avg_minutes': player_context.recent_avg_minutes or 0,
            'recent_avg_points': player_context.recent_avg_points or 0,
            'recent_avg_usage': player_context.recent_avg_usage or 0,
            'recent_avg_efg': player_context.recent_avg_efg or 0,
            'days_since_last_game': player_context.days_since_last_game or 0
        }
    
    def _get_opponent_context(self, player_id, opponent_team_id, game_date, lookback_days):
        """Get opponent context features"""
        query = text("""
            SELECT 
                COUNT(*) as games_vs_opponent,
                AVG(bs.points) as avg_points_vs_opponent,
                AVG(bs.minutes_played) as avg_minutes_vs_opponent,
                AVG(bs.field_goals_made) as avg_fgm_vs_opponent,
                AVG(bs.field_goals_attempted) as avg_fga_vs_opponent,
                AVG(bs.three_pointers_made) as avg_3pm_vs_opponent,
                AVG(bs.three_pointers_attempted) as avg_3pa_vs_opponent,
                AVG(bs.rebounds) as avg_rebounds_vs_opponent,
                AVG(bs.assists) as avg_assists_vs_opponent,
                AVG(bs.steals) as avg_steals_vs_opponent,
                AVG(bs.blocks) as avg_blocks_vs_opponent,
                AVG(bs.turnovers) as avg_turnovers_vs_opponent,
                AVG(abs.usg_pct) as avg_usage_vs_opponent,
                AVG(abs.efg_pct) as avg_efg_vs_opponent,
                AVG(abs.ts_pct) as avg_ts_vs_opponent,
                AVG(bsu.usg_pct) as avg_usage_pct_vs_opponent,
                AVG(bsu.pct_pts) as avg_pct_team_pts_vs_opponent,
                AVG(bsu.pct_fga) as avg_pct_team_fga_vs_opponent
            FROM nba_analytics.stg_box_scores bs
            LEFT JOIN nba_analytics.stg_advanced_box_scores abs ON bs.game_id = abs.game_id AND bs.player_id = abs.player_id
            LEFT JOIN nba_analytics.stg_box_score_usage bsu ON bs.game_id = bsu.game_id AND bs.player_id = bsu.player_id
            JOIN nba_analytics.stg_games gr ON bs.game_id = gr.game_id
            WHERE bs.player_id = :player_id
            AND gr.team_id = :opponent_team_id
            AND bs.game_date < :game_date
            AND bs.game_date >= :game_date - INTERVAL ':lookback_days days'
            AND bs.minutes_played > 0
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'opponent_team_id': opponent_team_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        return result.fetchone()
    
    def _extract_opponent_features(self, opponent_context):
        """Extract opponent features from context"""
        if not opponent_context:
            return {}
        
        return {
            'games_vs_opponent': opponent_context.games_vs_opponent or 0,
            'avg_points_vs_opponent': opponent_context.avg_points_vs_opponent or 0,
            'avg_minutes_vs_opponent': opponent_context.avg_minutes_vs_opponent or 0,
            'avg_fgm_vs_opponent': opponent_context.avg_fgm_vs_opponent or 0,
            'avg_fga_vs_opponent': opponent_context.avg_fga_vs_opponent or 0,
            'avg_3pm_vs_opponent': opponent_context.avg_3pm_vs_opponent or 0,
            'avg_3pa_vs_opponent': opponent_context.avg_3pa_vs_opponent or 0,
            'avg_rebounds_vs_opponent': opponent_context.avg_rebounds_vs_opponent or 0,
            'avg_assists_vs_opponent': opponent_context.avg_assists_vs_opponent or 0,
            'avg_steals_vs_opponent': opponent_context.avg_steals_vs_opponent or 0,
            'avg_blocks_vs_opponent': opponent_context.avg_blocks_vs_opponent or 0,
            'avg_turnovers_vs_opponent': opponent_context.avg_turnovers_vs_opponent or 0,
            'avg_usage_vs_opponent': opponent_context.avg_usage_vs_opponent or 0,
            'avg_efg_vs_opponent': opponent_context.avg_efg_vs_opponent or 0,
            'avg_ts_vs_opponent': opponent_context.avg_ts_vs_opponent or 0,
            'avg_usage_pct_vs_opponent': opponent_context.avg_usage_pct_vs_opponent or 0,
            'avg_pct_team_pts_vs_opponent': opponent_context.avg_pct_team_pts_vs_opponent or 0,
            'avg_pct_team_fga_vs_opponent': opponent_context.avg_pct_team_fga_vs_opponent or 0
        }
    
    def _get_team_context(self, team_id, game_date, lookback_days):
        """Get team context features"""
        query = text("""
            SELECT 
                COUNT(DISTINCT game_id) as total_games,
                AVG(active_players) as avg_active_players,
                AVG(dnp_players) as avg_dnp_players,
                AVG(injured_players) as avg_injured_players,
                AVG(avg_team_minutes) as avg_team_minutes
            FROM (
                SELECT 
                    bs.game_id,
                    COUNT(DISTINCT bs.player_id) as total_players,
                    COUNT(CASE WHEN bs.minutes_played > 0 THEN 1 END) as active_players,
                    COUNT(CASE WHEN bs.minutes_played = 0 AND bs.comments ILIKE '%DNP%' THEN 1 END) as dnp_players,
                    COUNT(CASE WHEN bs.comments ILIKE '%injury%' OR bs.comments ILIKE '%illness%' THEN 1 END) as injured_players,
                    AVG(bs.minutes_played) as avg_team_minutes
                FROM nba_analytics.stg_box_scores bs
                WHERE bs.team_id = :team_id
                AND bs.game_date < :game_date
                AND bs.game_date >= :game_date - INTERVAL ':lookback_days days'
                GROUP BY bs.game_id
            ) team_stats
        """)
        
        result = engine.execute(query, {
            'team_id': team_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        return result.fetchone()
    
    def _extract_team_features(self, team_context):
        """Extract team features from context"""
        if not team_context:
            return {}
        
        return {
            'team_total_games': team_context.total_games or 0,
            'team_avg_active_players': team_context.avg_active_players or 0,
            'team_avg_dnp_players': team_context.avg_dnp_players or 0,
            'team_avg_injured_players': team_context.avg_injured_players or 0,
            'team_avg_minutes': team_context.avg_team_minutes or 0
        }
    
    def _get_advanced_features(self, player_id, game_date, lookback_days):
        """Get advanced statistical features"""
        query = text("""
            SELECT 
                AVG(abs.usg_pct) as avg_usg_pct,
                AVG(abs.efg_pct) as avg_efg_pct,
                AVG(abs.ts_pct) as avg_ts_pct,
                AVG(abs.ast_pct) as avg_ast_pct,
                AVG(abs.reb_pct) as avg_reb_pct,
                AVG(abs.off_rating) as avg_off_rating,
                AVG(abs.def_rating) as avg_def_rating,
                AVG(abs.net_rating) as avg_net_rating,
                AVG(abs.pace) as avg_pace,
                AVG(abs.poss) as avg_poss
            FROM nba_analytics.stg_advanced_box_scores abs
            WHERE abs.player_id = :player_id
            AND abs.game_date < :game_date
            AND abs.game_date >= :game_date - INTERVAL ':lookback_days days'
            AND abs.minutes_played > 0
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        row = result.fetchone()
        if row:
            return {
                'avg_usg_pct': row.avg_usg_pct or 0,
                'avg_efg_pct': row.avg_efg_pct or 0,
                'avg_ts_pct': row.avg_ts_pct or 0,
                'avg_ast_pct': row.avg_ast_pct or 0,
                'avg_reb_pct': row.avg_reb_pct or 0,
                'avg_off_rating': row.avg_off_rating or 0,
                'avg_def_rating': row.avg_def_rating or 0,
                'avg_net_rating': row.avg_net_rating or 0,
                'avg_pace': row.avg_pace or 0,
                'avg_poss': row.avg_poss or 0
            }
        return {}
    
    def _get_usage_features(self, player_id, game_date, lookback_days):
        """Get usage percentage features"""
        query = text("""
            SELECT 
                AVG(usg_pct) as avg_usage_pct,
                AVG(pct_pts) as avg_pct_team_pts,
                AVG(pct_fga) as avg_pct_team_fga,
                AVG(pct_ast) as avg_pct_team_ast,
                AVG(pct_reb) as avg_pct_team_reb,
                AVG(pct_tov) as avg_pct_team_tov,
                AVG(pct_stl) as avg_pct_team_stl,
                AVG(pct_blk) as avg_pct_team_blk
            FROM nba_analytics.stg_box_score_usage
            WHERE player_id = :player_id
            AND game_date < :game_date
            AND game_date >= :game_date - INTERVAL ':lookback_days days'
            AND minutes_played > 0
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        row = result.fetchone()
        if row:
            return {
                'avg_usage_pct': row.avg_usage_pct or 0,
                'avg_pct_team_pts': row.avg_pct_team_pts or 0,
                'avg_pct_team_fga': row.avg_pct_team_fga or 0,
                'avg_pct_team_ast': row.avg_pct_team_ast or 0,
                'avg_pct_team_reb': row.avg_pct_team_reb or 0,
                'avg_pct_team_tov': row.avg_pct_team_tov or 0,
                'avg_pct_team_stl': row.avg_pct_team_stl or 0,
                'avg_pct_team_blk': row.avg_pct_team_blk or 0
            }
        return {}
    
    def _get_shot_features(self, player_id, game_date, lookback_days):
        """Get shot location features"""
        query = text("""
            SELECT 
                COUNT(*) as total_shots,
                COUNT(CASE WHEN shot_made_flag = 1 THEN 1 END) as shots_made,
                COUNT(CASE WHEN shot_made_flag = 0 THEN 1 END) as shots_missed,
                AVG(shot_distance) as avg_shot_distance,
                COUNT(CASE WHEN shot_zone_basic = 'Restricted Area' THEN 1 END) as restricted_area_shots,
                COUNT(CASE WHEN shot_zone_basic = 'In The Paint (Non-RA)' THEN 1 END) as paint_shots,
                COUNT(CASE WHEN shot_zone_basic = 'Mid-Range' THEN 1 END) as mid_range_shots,
                COUNT(CASE WHEN shot_zone_basic = 'Left Corner 3' THEN 1 END) as left_corner_3_shots,
                COUNT(CASE WHEN shot_zone_basic = 'Right Corner 3' THEN 1 END) as right_corner_3_shots,
                COUNT(CASE WHEN shot_zone_basic = 'Above the Break 3' THEN 1 END) as above_break_3_shots
            FROM nba_analytics.stg_player_shot_locations
            WHERE player_id = :player_id
            AND game_date < :game_date
            AND game_date >= :game_date - INTERVAL ':lookback_days days'
        """)
        
        result = engine.execute(query, {
            'player_id': player_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        row = result.fetchone()
        if row:
            total_shots = row.total_shots or 0
            return {
                'total_shots': total_shots,
                'shots_made': row.shots_made or 0,
                'shots_missed': row.shots_missed or 0,
                'shot_fg_pct': (row.shots_made or 0) / max(1, total_shots),
                'avg_shot_distance': row.avg_shot_distance or 0,
                'restricted_area_shots': row.restricted_area_shots or 0,
                'paint_shots': row.paint_shots or 0,
                'mid_range_shots': row.mid_range_shots or 0,
                'left_corner_3_shots': row.left_corner_3_shots or 0,
                'right_corner_3_shots': row.right_corner_3_shots or 0,
                'above_break_3_shots': row.above_break_3_shots or 0
            }
        return {}
    
    def save_features_to_database(self, features_list):
        """
        Save the feature set to the database
        """
        if not features_list:
            print("No features to save")
            return
        
        # Create the features table if it doesn't exist
        self._create_features_table()
        
        # Insert features
        with engine.begin() as conn:
            for features in features_list:
                # Remove None values and convert to proper types
                clean_features = {k: v for k, v in features.items() if v is not None}
                
                # Build insert statement dynamically
                columns = list(clean_features.keys())
                values = list(clean_features.values())
                placeholders = [f':{col}' for col in columns]
                
                insert_stmt = text(f"""
                    INSERT INTO nba_analytics.player_game_features (
                        {', '.join(columns)}
                    ) VALUES (
                        {', '.join(placeholders)}
                    )
                    ON CONFLICT (player_id, game_id) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP
                """)
                
                conn.execute(insert_stmt, clean_features)
        
        print(f"Saved {len(features_list)} feature records to database")
    
    def _create_features_table(self):
        """Create the player game features table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS nba_analytics.player_game_features (
            player_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            game_id VARCHAR(20) NOT NULL,
            opponent_team_id INTEGER,
            game_date DATE,
            
            -- Player context features
            total_games INTEGER,
            games_played INTEGER,
            dnp_games INTEGER,
            avg_minutes DECIMAL(10,2),
            avg_points DECIMAL(10,2),
            avg_fgm DECIMAL(10,2),
            avg_fga DECIMAL(10,2),
            avg_3pm DECIMAL(10,2),
            avg_3pa DECIMAL(10,2),
            avg_rebounds DECIMAL(10,2),
            avg_assists DECIMAL(10,2),
            avg_steals DECIMAL(10,2),
            avg_blocks DECIMAL(10,2),
            avg_turnovers DECIMAL(10,2),
            avg_usage DECIMAL(10,2),
            avg_efg DECIMAL(10,2),
            avg_ts DECIMAL(10,2),
            avg_ast_pct DECIMAL(10,2),
            avg_reb_pct DECIMAL(10,2),
            avg_off_rating DECIMAL(10,2),
            avg_def_rating DECIMAL(10,2),
            avg_net_rating DECIMAL(10,2),
            avg_usage_pct DECIMAL(10,2),
            avg_pct_team_pts DECIMAL(10,2),
            avg_pct_team_fga DECIMAL(10,2),
            avg_pct_team_ast DECIMAL(10,2),
            avg_pct_team_reb DECIMAL(10,2),
            recent_avg_minutes DECIMAL(10,2),
            recent_avg_points DECIMAL(10,2),
            recent_avg_usage DECIMAL(10,2),
            recent_avg_efg DECIMAL(10,2),
            days_since_last_game INTEGER,
            
            -- Opponent features
            games_vs_opponent INTEGER,
            avg_points_vs_opponent DECIMAL(10,2),
            avg_minutes_vs_opponent DECIMAL(10,2),
            avg_fgm_vs_opponent DECIMAL(10,2),
            avg_fga_vs_opponent DECIMAL(10,2),
            avg_3pm_vs_opponent DECIMAL(10,2),
            avg_3pa_vs_opponent DECIMAL(10,2),
            avg_rebounds_vs_opponent DECIMAL(10,2),
            avg_assists_vs_opponent DECIMAL(10,2),
            avg_steals_vs_opponent DECIMAL(10,2),
            avg_blocks_vs_opponent DECIMAL(10,2),
            avg_turnovers_vs_opponent DECIMAL(10,2),
            avg_usage_vs_opponent DECIMAL(10,2),
            avg_efg_vs_opponent DECIMAL(10,2),
            avg_ts_vs_opponent DECIMAL(10,2),
            avg_usage_pct_vs_opponent DECIMAL(10,2),
            avg_pct_team_pts_vs_opponent DECIMAL(10,2),
            avg_pct_team_fga_vs_opponent DECIMAL(10,2),
            
            -- Team features
            team_total_games INTEGER,
            team_avg_active_players DECIMAL(10,2),
            team_avg_dnp_players DECIMAL(10,2),
            team_avg_injured_players DECIMAL(10,2),
            team_avg_minutes DECIMAL(10,2),
            
            -- Advanced features
            avg_usg_pct DECIMAL(10,2),
            avg_efg_pct DECIMAL(10,2),
            avg_ts_pct DECIMAL(10,2),
            avg_ast_pct DECIMAL(10,2),
            avg_reb_pct DECIMAL(10,2),
            avg_off_rating DECIMAL(10,2),
            avg_def_rating DECIMAL(10,2),
            avg_net_rating DECIMAL(10,2),
            avg_pace DECIMAL(10,2),
            avg_poss DECIMAL(10,2),
            
            -- Usage features
            avg_usage_pct DECIMAL(10,2),
            avg_pct_team_pts DECIMAL(10,2),
            avg_pct_team_fga DECIMAL(10,2),
            avg_pct_team_ast DECIMAL(10,2),
            avg_pct_team_reb DECIMAL(10,2),
            avg_pct_team_tov DECIMAL(10,2),
            avg_pct_team_stl DECIMAL(10,2),
            avg_pct_team_blk DECIMAL(10,2),
            
            -- Shot features
            total_shots INTEGER,
            shots_made INTEGER,
            shots_missed INTEGER,
            shot_fg_pct DECIMAL(10,2),
            avg_shot_distance DECIMAL(10,2),
            restricted_area_shots INTEGER,
            paint_shots INTEGER,
            mid_range_shots INTEGER,
            left_corner_3_shots INTEGER,
            right_corner_3_shots INTEGER,
            above_break_3_shots INTEGER,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            PRIMARY KEY (player_id, game_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_player_game_features_player_id ON nba_analytics.player_game_features(player_id);
        CREATE INDEX IF NOT EXISTS idx_player_game_features_game_id ON nba_analytics.player_game_features(game_id);
        CREATE INDEX IF NOT EXISTS idx_player_game_features_game_date ON nba_analytics.player_game_features(game_date);
        """
        
        engine.execute(text(create_table_sql))

# Example usage
if __name__ == "__main__":
    feature_engineer = FeatureEngineeringSystem()
    
    # Build features for a specific game date
    game_date = datetime(2024, 12, 15).date()
    features = feature_engineer.build_player_game_features(game_date)
    
    # Save to database
    feature_engineer.save_features_to_database(features)
    
    print(f"Built and saved {len(features)} feature records for {game_date}") 