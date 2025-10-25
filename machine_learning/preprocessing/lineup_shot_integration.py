#!/usr/bin/env python3
"""
Lineup Shot Pattern Integration for Player Prediction
Uses preprocessed database data instead of direct API calls
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

class LineupShotIntegration:
    def __init__(self):
        self.engine = engine
        
    def get_available_lineups_from_db(self, season="2024-25", team_id=None):
        """
        Get available lineups from database instead of API
        """
        query = text("""
            SELECT 
                group_id,
                group_name,
                team_id,
                team_name,
                season,
                games_played,
                wins,
                losses,
                minutes_played,
                off_rating,
                def_rating,
                net_rating,
                ast_pct,
                ast_to,
                ast_ratio,
                oreb_pct,
                dreb_pct,
                reb_pct,
                tm_tov_pct,
                efg_pct,
                ts_pct,
                usg_pct,
                e_usg_pct,
                e_pace,
                pace,
                pace_per40,
                poss,
                pie
            FROM nba_analytics.stg_lineups
            WHERE season = :season
            AND (:team_id IS NULL OR team_id = :team_id)
            ORDER BY games_played DESC
        """)
        
        result = engine.execute(query, {
            'season': season,
            'team_id': team_id
        })
        
        return [dict(row) for row in result]
    
    def get_lineup_shot_data_from_db(self, group_id, season="2024-25"):
        """
        Get shot data for a specific lineup from database
        """
        query = text("""
            SELECT 
                group_id,
                season,
                game_id,
                game_event_id,
                player_id,
                player_name,
                team_id,
                team_name,
                period,
                minutes_remaining,
                seconds_remaining,
                event_type,
                action_type,
                shot_type,
                shot_zone_basic,
                shot_zone_area,
                shot_zone_range,
                shot_distance,
                loc_x,
                loc_y,
                shot_attempted_flag,
                shot_made_flag,
                game_date,
                home_team,
                away_team
            FROM nba_analytics.stg_lineup_shot_data
            WHERE group_id = :group_id
            AND season = :season
            ORDER BY game_date DESC
        """)
        
        result = engine.execute(query, {
            'group_id': group_id,
            'season': season
        })
        
        return [dict(row) for row in result]
    
    def analyze_lineup_shot_patterns(self, lineup_data):
        """
        Analyze shot patterns from lineup data
        """
        if not lineup_data:
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(lineup_data)
        
        # Basic shot pattern analysis
        analysis = {
            'total_shots': len(df),
            'total_made': len(df[df['shot_made_flag'] == 1]),
            'total_missed': len(df[df['shot_made_flag'] == 0]),
            'fg_pct': len(df[df['shot_made_flag'] == 1]) / len(df) if len(df) > 0 else 0,
            
            # Shot zone analysis
            'shot_zones': df['shot_zone_basic'].value_counts().to_dict(),
            'shot_areas': df['shot_zone_area'].value_counts().to_dict(),
            'shot_ranges': df['shot_zone_range'].value_counts().to_dict(),
            
            # Distance analysis
            'avg_shot_distance': df['shot_distance'].mean() if 'shot_distance' in df.columns else None,
            'shot_distances': df['shot_distance'].value_counts().to_dict() if 'shot_distance' in df.columns else {},
            
            # Player analysis
            'player_shots': df['player_name'].value_counts().to_dict(),
            'player_made': df[df['shot_made_flag'] == 1]['player_name'].value_counts().to_dict(),
            
            # Game analysis
            'games_represented': df['game_id'].nunique(),
            'unique_games': df['game_id'].unique().tolist()
        }
        
        return analysis
    
    def get_player_lineup_shot_profile(self, player_id, team_id, season="2024-25"):
        """
        Get shot profile for a player across all lineups they're part of
        """
        # Get all lineups for the team from database
        team_lineups = self.get_available_lineups_from_db(season, team_id)
        
        player_shot_profiles = []
        
        for lineup in team_lineups:
            group_id = lineup['group_id']
            group_name = lineup['group_name']
            
            # Check if this lineup includes our player
            if str(player_id) in group_id:
                # Get shot data for this lineup from database
                lineup_data = self.get_lineup_shot_data_from_db(group_id, season)
                
                if lineup_data:
                    # Filter shots for this specific player
                    player_shots = []
                    for shot in lineup_data:
                        if shot.get('player_id') == player_id:
                            player_shots.append(shot)
                    
                    if player_shots:
                        # Analyze this player's shots in this lineup
                        df = pd.DataFrame(player_shots)
                        
                        profile = {
                            'lineup_id': group_id,
                            'lineup_name': group_name,
                            'total_shots': len(df),
                            'shots_made': len(df[df['shot_made_flag'] == 1]),
                            'shots_missed': len(df[df['shot_made_flag'] == 0]),
                            'fg_pct': len(df[df['shot_made_flag'] == 1]) / len(df) if len(df) > 0 else 0,
                            
                            # Shot distribution
                            'shot_zones': df['shot_zone_basic'].value_counts().to_dict(),
                            'shot_areas': df['shot_zone_area'].value_counts().to_dict(),
                            'shot_ranges': df['shot_zone_range'].value_counts().to_dict(),
                            
                            # Distance analysis
                            'avg_distance': df['shot_distance'].mean() if 'shot_distance' in df.columns else None,
                            'distance_distribution': df['shot_distance'].value_counts().to_dict() if 'shot_distance' in df.columns else {},
                            
                            # Shot types
                            'shot_types': df['shot_type'].value_counts().to_dict(),
                            'action_types': df['action_type'].value_counts().to_dict(),
                            
                            # Games played in this lineup
                            'games_in_lineup': df['game_id'].nunique(),
                            'game_ids': df['game_id'].unique().tolist()
                        }
                        
                        player_shot_profiles.append(profile)
        
        return player_shot_profiles
    
    def predict_player_shots_for_lineup(self, player_id, team_id, opponent_team_id, game_date, season="2024-25"):
        """
        Predict shot patterns for a player in an upcoming game based on lineup context
        """
        # Get player's historical shot profiles across different lineups
        shot_profiles = self.get_player_lineup_shot_profile(player_id, team_id, season)
        
        if not shot_profiles:
            return None
        
        # Calculate weighted averages based on recent performance
        predictions = {
            'expected_shots': 0,
            'expected_fg_pct': 0,
            'expected_shot_distribution': {},
            'expected_distance': 0,
            'confidence_score': 0
        }
        
        total_weight = 0
        
        for profile in shot_profiles:
            # Calculate weight based on recency and sample size
            weight = profile['total_shots'] * profile['games_in_lineup']
            
            if weight > 0:
                predictions['expected_shots'] += (profile['total_shots'] / profile['games_in_lineup']) * weight
                predictions['expected_fg_pct'] += profile['fg_pct'] * weight
                predictions['expected_distance'] += (profile['avg_distance'] or 0) * weight
                total_weight += weight
                
                # Aggregate shot distributions
                for zone, count in profile['shot_zones'].items():
                    if zone not in predictions['expected_shot_distribution']:
                        predictions['expected_shot_distribution'][zone] = 0
                    predictions['expected_shot_distribution'][zone] += count * weight
        
        # Normalize by total weight
        if total_weight > 0:
            predictions['expected_shots'] /= total_weight
            predictions['expected_fg_pct'] /= total_weight
            predictions['expected_distance'] /= total_weight
            
            # Normalize shot distribution
            for zone in predictions['expected_shot_distribution']:
                predictions['expected_shot_distribution'][zone] /= total_weight
            
            # Calculate confidence score based on data quality
            predictions['confidence_score'] = min(1.0, total_weight / 100)  # Normalize to 0-1
        
        return predictions
    
    def get_lineup_availability_for_game(self, team_id, game_date, lookback_days=30):
        """
        Predict which lineups are likely to be available for an upcoming game
        """
        query = text("""
            WITH recent_lineup_usage AS (
                SELECT 
                    bs.game_id,
                    bs.game_date,
                    bs.team_id,
                    -- Create lineup identifier (simplified)
                    STRING_AGG(DISTINCT bs.player_id::TEXT, '-' ORDER BY bs.player_id) as lineup_id,
                    COUNT(DISTINCT bs.player_id) as players_in_lineup,
                    AVG(bs.minutes_played) as avg_minutes_in_lineup,
                    COUNT(*) as lineup_occurrences
                FROM nba_analytics.stg_box_scores bs
                WHERE bs.team_id = :team_id
                AND bs.game_date < :game_date
                AND bs.game_date >= :game_date - INTERVAL ':lookback_days days'
                AND bs.minutes_played > 0
                GROUP BY bs.game_id, bs.game_date, bs.team_id
                HAVING COUNT(DISTINCT bs.player_id) = 5  -- Only 5-player lineups
            )
            SELECT 
                lineup_id,
                COUNT(DISTINCT game_id) as games_used,
                AVG(avg_minutes_in_lineup) as avg_minutes_per_player,
                MAX(game_date) as last_used_date,
                -- Recent usage trend
                COUNT(CASE WHEN game_date >= :game_date - INTERVAL '7 days' THEN 1 END) as recent_usage
            FROM recent_lineup_usage
            GROUP BY lineup_id
            ORDER BY recent_usage DESC, games_used DESC
        """)
        
        result = engine.execute(query, {
            'team_id': team_id,
            'game_date': game_date,
            'lookback_days': lookback_days
        })
        
        return result.fetchall()
    
    def build_comprehensive_prediction(self, player_id, team_id, opponent_team_id, game_date, season="2024-25"):
        """
        Build comprehensive prediction combining all data sources
        """
        prediction = {
            'player_id': player_id,
            'team_id': team_id,
            'opponent_team_id': opponent_team_id,
            'game_date': game_date,
            'predictions': {}
        }
        
        # 1. Get lineup availability prediction
        available_lineups = self.get_lineup_availability_for_game(team_id, game_date)
        
        # 2. Get player's shot predictions for each likely lineup
        for lineup in available_lineups[:5]:  # Top 5 most likely lineups
            lineup_id = lineup.lineup_id
            
            # Check if this lineup includes our player
            if str(player_id) in lineup_id:
                shot_prediction = self.predict_player_shots_for_lineup(
                    player_id, team_id, opponent_team_id, game_date, season
                )
                
                if shot_prediction:
                    prediction['predictions'][lineup_id] = {
                        'lineup_usage_probability': lineup.recent_usage / max(1, lineup.games_used),
                        'shot_prediction': shot_prediction
                    }
        
        # 3. Calculate weighted average prediction
        if prediction['predictions']:
            total_weight = 0
            weighted_shots = 0
            weighted_fg_pct = 0
            weighted_distance = 0
            
            for lineup_id, lineup_pred in prediction['predictions'].items():
                weight = lineup_pred['lineup_usage_probability'] * lineup_pred['shot_prediction']['confidence_score']
                
                weighted_shots += lineup_pred['shot_prediction']['expected_shots'] * weight
                weighted_fg_pct += lineup_pred['shot_prediction']['expected_fg_pct'] * weight
                weighted_distance += lineup_pred['shot_prediction']['expected_distance'] * weight
                total_weight += weight
            
            if total_weight > 0:
                prediction['final_prediction'] = {
                    'expected_shots': weighted_shots / total_weight,
                    'expected_fg_pct': weighted_fg_pct / total_weight,
                    'expected_distance': weighted_distance / total_weight,
                    'confidence_score': total_weight / len(prediction['predictions'])
                }
        
        return prediction

# Example usage
if __name__ == "__main__":
    integrator = LineupShotIntegration()
    
    # Example: Predict shots for a player in an upcoming game
    prediction = integrator.build_comprehensive_prediction(
        player_id=1626157,  # Example player ID
        team_id=1610612752,  # Example team ID
        opponent_team_id=1610612755,  # Example opponent
        game_date=datetime(2024, 12, 15),  # Example game date
        season="2024-25"
    )
    
    print("Comprehensive Player Prediction:")
    print(f"Player ID: {prediction['player_id']}")
    print(f"Game Date: {prediction['game_date']}")
    
    if 'final_prediction' in prediction:
        final = prediction['final_prediction']
        print(f"Expected Shots: {final['expected_shots']:.1f}")
        print(f"Expected FG%: {final['expected_fg_pct']:.3f}")
        print(f"Expected Distance: {final['expected_distance']:.1f} ft")
        print(f"Confidence Score: {final['confidence_score']:.3f}")
    
    print(f"\nLineup-specific predictions: {len(prediction['predictions'])} lineups analyzed") 