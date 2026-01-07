"""
FIXED feature engineering pipeline that works with your actual database structure
ENHANCED VERSION with player status features
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class FixedFeaturePipeline:
    def __init__(self, db=None):
        """
        Initialize with database connection.
        """
        self.db = db
        self.rolling_windows = [3,5,10]
        
    def generate_features(self, gameweek: int, season: str = "2025-2026") -> pd.DataFrame:
        """
        Generate ENHANCED features using ACTUAL database structure
        """
        print(f"\n{'='*60}")
        print(f"Generating ENHANCED features for Gameweek {gameweek}, {season}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Get player history with ALL available columns
            print("1. Loading player history...")
            player_history = self._get_player_history_complete(gameweek, season)
            
            if player_history.empty:
                print(" No player history found.")
                return pd.DataFrame()
            
            print(f" Loaded {len(player_history)} player-gameweek records with {len(player_history.columns)} columns")
            
            # Step 2: Calculate rolling features
            print("2. Calculating rolling features...")
            features_df = self._calculate_rolling_features(player_history)
            
            # Step 3: Add player metadata
            print("3. Adding player metadata...")
            features_df = self._add_player_metadata(features_df, season)
            
            # Step 4: NEW - Add player status features (CRITICAL)
            print("4. Adding player status and availability features...")
            features_df = self._add_player_status_features_enhanced(features_df, gameweek, season)
            
            # Step 5: Calculate value metrics
            print("5. Calculating value metrics...")
            features_df = self._calculate_value_metrics(features_df)
            
            # Step 6: Add target variable
            print("6. Adding target variable...")
            features_df = self._add_target_variable(features_df, gameweek, season)
            
            # Step 7: Add match context from matches table
            print("7. Adding match context...")
            features_df = self._add_match_context(features_df, season)
            
            # Step 8: Add team strength metrics
            print("8. Adding team strength metrics...")
            features_df = self._add_team_strength_metrics_fixed(features_df, season)
            
            # Step 9: Add xG-based metrics
            print("9. Adding xG-based metrics...")
            features_df = self._add_xg_based_metrics_fixed(features_df)
            
            # Step 10: Add fixture difficulty using matches data
            print("10. Adding fixture difficulty...")
            features_df = self._add_fixture_difficulty_fixed(features_df, gameweek, season)
            
            # Step 11: Clean and finalize features
            print("11. Cleaning and finalizing features...")
            features_df = self._clean_features_enhanced(features_df)
            
            print(f"\n✓ ENHANCED feature generation complete!")
            print(f" Final dataset: {len(features_df)} records, {len(features_df.columns)} features")
            
            # Show feature categories
            self._print_enhanced_feature_categories(features_df)
            
            return features_df
            
        except Exception as e:
            print(f"✗ Error generating features: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _add_player_status_features_enhanced(self, features_df: pd.DataFrame, 
                                           gameweek: int, 
                                           season: str) -> pd.DataFrame:
        """
        NEW: Add player status and availability features from player_stats table
        """
        if features_df.empty or not self.db:
            return features_df
        
        try:
            # Get player status data for current gameweek
            query = """
                SELECT 
                    id as player_id,
                    status,
                    chance_of_playing_next_round,
                    chance_of_playing_this_round,
                    selected_by_percent,
                    form,
                    points_per_game,
                    starts,
                    expected_goals,
                    expected_assists,
                    expected_goals_conceded,
                    expected_goals_per_90,
                    expected_assists_per_90,
                    expected_goals_conceded_per_90,
                    influence,
                    creativity,
                    threat,
                    ict_index,
                    bps,
                    transfers_in,
                    transfers_out,
                    transfers_in_event,
                    transfers_out_event,
                    value_form,
                    value_season,
                    ep_next,
                    ep_this,
                    now_cost,
                    cost_change_event
                FROM player_stats 
                WHERE gw = ? 
                  AND season = ?
            """
            
            status_data = self.db.execute_query(query, (gameweek, season))
            
            if status_data is None or status_data.empty:
                print(f" ⚠ No player status data found for GW{gameweek}")
                return features_df
            
            # Standardize player_id type
            features_df['player_id'] = features_df['player_id'].astype(str)
            status_data['player_id'] = status_data['player_id'].astype(str)
            
            # Merge status data with features
            result_df = features_df.merge(
                status_data,
                on='player_id',
                how='left',
                suffixes=('', '_status')
            )
            
            # Process status features
            # 1. Convert status to numeric
            status_mapping = {'a': 1, 'i': 0, 'u': 0, 'd': 0.5}
            result_df['status_numeric'] = result_df['status'].map(status_mapping).fillna(0)
            
            # 2. Process playing chances
            result_df['playing_chance_next'] = result_df['chance_of_playing_next_round'].fillna(0) / 100.0
            result_df['playing_chance_this'] = result_df['chance_of_playing_this_round'].fillna(0) / 100.0
            
            # Composite playing probability
            result_df['playing_probability'] = (
                result_df['playing_chance_next'].fillna(0.5) * 0.6 + 
                result_df['playing_chance_this'].fillna(0.5) * 0.4
            )
            
            # 3. Ownership features
            result_df['ownership_pct'] = result_df['selected_by_percent'].fillna(0) / 100.0
            result_df['high_ownership'] = (result_df['ownership_pct'] > 0.2).astype(int)
            
            # 4. Form features
            result_df['form_adjusted'] = result_df['form'].fillna(0) * result_df['playing_probability']
            result_df['ppg_adjusted'] = result_df['points_per_game'].fillna(0) * result_df['playing_probability']
            
            # 5. Transfer momentum
            result_df['transfer_balance'] = (
                result_df['transfers_in'].fillna(0) - result_df['transfers_out'].fillna(0)
            )
            result_df['transfer_net'] = (
                result_df['transfers_in_event'].fillna(0) - result_df['transfers_out_event'].fillna(0)
            )
            
            # 6. Value metrics
            result_df['value_form'] = result_df['value_form'].fillna(0)
            result_df['value_season'] = result_df['value_season'].fillna(0)
            
            # 7. Expected points
            result_df['ep_next'] = result_df['ep_next'].fillna(0)
            result_df['ep_this'] = result_df['ep_this'].fillna(0)
            result_df['ep_total'] = result_df['ep_next'] + result_df['ep_this']
            result_df['ep_total_adj'] = result_df['ep_total'] * result_df['playing_probability']
            
            # 8. ICT Index
            result_df['influence'] = result_df['influence'].fillna(0)
            result_df['creativity'] = result_df['creativity'].fillna(0)
            result_df['threat'] = result_df['threat'].fillna(0)
            result_df['ict_index'] = result_df['ict_index'].fillna(0)
            result_df['bps'] = result_df['bps'].fillna(0)
            
            # 9. Cost change
            result_df['cost_change_trend'] = result_df['cost_change_event'].fillna(0)
            result_df['cost_increasing'] = (result_df['cost_change_trend'] > 0).astype(int)
            
            # 10. Add position-specific status features
            result_df = self._add_position_specific_status_features(result_df)
            
            print(f" Added {len(status_data.columns)} status features for {len(status_data)} players")
            
            return result_df
            
        except Exception as e:
            print(f" ⚠ Error adding player status features: {e}")
            return features_df
    
    def _add_position_specific_status_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific status features"""
        if features_df.empty or 'position' not in features_df.columns:
            return features_df
        
        result_df = features_df.copy()
        
        # Goalkeeper specific
        if 'expected_goals_conceded' in result_df.columns and 'minutes' in result_df.columns:
            goalkeepers = result_df[result_df['position'] == 'Goalkeeper'].copy()
            if not goalkeepers.empty:
                result_df.loc[goalkeepers.index, 'xgc_per_90'] = goalkeepers.apply(
                    lambda row: row['expected_goals_conceded'] * 90 / row['minutes'] if row['minutes'] > 0 else 0,
                    axis=1
                )
                result_df.loc[goalkeepers.index, 'xgc_per_90_adj'] = (
                    result_df.loc[goalkeepers.index, 'xgc_per_90'] * 
                    result_df.loc[goalkeepers.index, 'playing_probability']
                )
        
        # Defender specific
        defenders = result_df[result_df['position'] == 'Defender'].copy()
        if not defenders.empty and 'expected_goals_conceded' in defenders.columns:
            result_df.loc[defenders.index, 'clean_sheet_probability'] = np.exp(-defenders['expected_goals_conceded'].fillna(0))
            result_df.loc[defenders.index, 'clean_sheet_probability_adj'] = (
                result_df.loc[defenders.index, 'clean_sheet_probability'] * 
                result_df.loc[defenders.index, 'playing_probability']
            )
        
        # Forward specific
        forwards = result_df[result_df['position'] == 'Forward'].copy()
        if not forwards.empty and 'expected_goals' in forwards.columns:
            result_df.loc[forwards.index, 'goal_scoring_probability'] = 1 - np.exp(-forwards['expected_goals'].fillna(0))
            result_df.loc[forwards.index, 'goal_scoring_probability_adj'] = (
                result_df.loc[forwards.index, 'goal_scoring_probability'] * 
                result_df.loc[forwards.index, 'playing_probability']
            )
        
        return result_df
    
    def _clean_features_enhanced(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize ENHANCED features"""
        if features_df.empty:
            return features_df
            
        cleaned_df = features_df.copy()
        
        # Keep only the latest gameweek for each player
        if 'gameweek' in cleaned_df.columns:
            latest_rows = cleaned_df.sort_values('gameweek').groupby('player_id').tail(1)
        else:
            latest_rows = cleaned_df
        
        # Fill NaN values with appropriate defaults for ENHANCED features
        for col in latest_rows.columns:
            if col.startswith('rolling_') or col.endswith('_score'):
                latest_rows[col] = latest_rows[col].fillna(0)
            elif col.endswith('_efficiency'):
                latest_rows[col] = latest_rows[col].fillna(1.0)
            elif col.endswith('_overperformance'):
                latest_rows[col] = latest_rows[col].fillna(0)
            elif col.endswith('_fdr'):
                latest_rows[col] = latest_rows[col].fillna(3.0)
            elif col.endswith('_probability') or col.endswith('_chance'):
                latest_rows[col] = latest_rows[col].fillna(0.5)
            elif col in ['strength', 'team_strength_composite']:
                latest_rows[col] = latest_rows[col].fillna(3.0)
            elif col == 'elo':
                latest_rows[col] = latest_rows[col].fillna(1500.0)
            elif col == 'actual_points':
                latest_rows[col] = latest_rows[col].fillna(0)
            elif col in ['selected_by_percent', 'form', 'points_per_game']:
                latest_rows[col] = latest_rows[col].fillna(0)
            elif col in ['playing_probability', 'ownership_pct', 'form_adjusted']:
                latest_rows[col] = latest_rows[col].fillna(0)
            elif col in ['status_numeric']:
                latest_rows[col] = latest_rows[col].fillna(0)
            elif col in ['ep_total', 'ep_total_adj']:
                latest_rows[col] = latest_rows[col].fillna(0)
            elif col in ['influence', 'creativity', 'threat', 'ict_index', 'bps']:
                latest_rows[col] = latest_rows[col].fillna(0)
        
        # Add metadata
        latest_rows['feature_generation_time'] = datetime.now()
        latest_rows['season'] = "2025-2026"
        latest_rows['feature_version'] = 'enhanced_v1.0'
        
        print(f" Final cleaned dataset: {len(latest_rows)} records")
        
        # Verify critical features are present
        critical_features = ['selected_by_percent', 'form', 'playing_probability']
        for feature in critical_features:
            if feature in latest_rows.columns:
                non_null = latest_rows[feature].notnull().sum()
                print(f"   {feature}: {non_null}/{len(latest_rows)} non-null")
        
        return latest_rows
    
    def _print_enhanced_feature_categories(self, features_df: pd.DataFrame):
        """Print ENHANCED feature categories summary"""
        if features_df.empty:
            return
        
        # Define ENHANCED feature categories
        categories = {
            'Availability & Status': [col for col in features_df.columns 
                                     if any(keyword in col for keyword in 
                                           ['status', 'playing', 'chance', 'probability'])],
            'Form & Performance': [col for col in features_df.columns 
                                  if any(keyword in col for keyword in 
                                        ['form', 'points_per_game', 'bps', 'ict', 'influence', 'creativity', 'threat'])],
            'Ownership & Market': [col for col in features_df.columns 
                                  if any(keyword in col for keyword in 
                                        ['selected', 'ownership', 'transfer', 'cost_change', 'value'])],
            'Expected Metrics': [col for col in features_df.columns 
                                if any(keyword in col for keyword in 
                                      ['expected', 'xg', 'xa', 'xgc', 'ep_'])],
            'Rolling Performance': [col for col in features_df.columns 
                                   if 'rolling_' in col],
            'Team Strength': [col for col in features_df.columns 
                            if any(keyword in col for keyword in 
                                  ['strength', 'elo', 'team_'])],
            'Fixture Difficulty': [col for col in features_df.columns 
                                 if 'fdr' in col.lower()],
            'Value Metrics': [col for col in features_df.columns 
                            if any(keyword in col for keyword in 
                                  ['cost', 'million', 'value', 'per_cost'])],
            'Basic Stats': ['minutes', 'total_points', 'goals_scored', 'assists', 
                          'clean_sheets', 'goals_conceded', 'bonus', 'saves']
        }
        
        print("\nENHANCED Feature Categories Summary:")
        for category, features in categories.items():
            existing_features = [f for f in features if f in features_df.columns]
            if existing_features:
                print(f" {category}: {len(existing_features)} features")
    
    
    def _get_player_history_complete(self, gameweek: int, season: str) -> pd.DataFrame:
        """Get player history with ALL available columns"""
        try:
            if not self.db:
                return pd.DataFrame()
            
            query = """
            SELECT 
                id as player_id,
                gw as gameweek,
                season,
                minutes,
                total_points,
                goals_scored,
                assists,
                clean_sheets,
                goals_conceded,
                expected_goals,
                expected_assists,
                expected_goal_involvements,
                expected_goals_conceded,
                saves,
                bonus,
                selected_by_percent,
                now_cost,
                tackles,
                clearances_blocks_interceptions,
                influence,
                creativity,
                threat,
                ict_index,
                bps,
                yellow_cards,
                red_cards,
                own_goals,
                penalties_saved,
                penalties_missed,
                form
            FROM player_gameweek_stats
            WHERE gw < ?
              AND season = ?
              AND minutes > 0
            ORDER BY player_id, gameweek
            """
            
            history_data = self.db.execute_query(query, (gameweek, season))
            
            if history_data is None or history_data.empty:
                return pd.DataFrame()
            
            print(f" Successfully loaded {len(history_data)} records")
            print(f" Available columns: {list(history_data.columns)}")
            
            return history_data
            
        except Exception as e:
            print(f" Error in _get_player_history_complete: {e}")
            return pd.DataFrame()
    
    def _calculate_rolling_features(self, player_history: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages for each player"""
        if player_history.empty:
            return player_history
            
        # Sort by player and gameweek
        df = player_history.sort_values(['player_id', 'gameweek']).copy()
        
        # Initialize result DataFrame
        result_df = df.copy()
        
        # Group by player to calculate rolling stats
        for player_id, group in df.groupby('player_id'):
            group_sorted = group.sort_values('gameweek')
            
            # Calculate rolling features for each window
            for window in self.rolling_windows:
                if len(group_sorted) >= window:
                    # Points
                    if 'total_points' in group_sorted.columns:
                        col_name = f'rolling_points_{window}'
                        roll_values = group_sorted['total_points'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                    
                    # Minutes
                    if 'minutes' in group_sorted.columns:
                        col_name = f'rolling_minutes_{window}'
                        roll_values = group_sorted['minutes'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                        
                        # Minutes consistency
                        col_name = f'minutes_consistency_{window}'
                        minutes_bool = (group_sorted['minutes'] > 60).astype(int)
                        consistency = minutes_bool.rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = consistency.values
                    
                    # Expected goals
                    if 'expected_goals' in group_sorted.columns:
                        col_name = f'rolling_xg_{window}'
                        roll_values = group_sorted['expected_goals'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                    
                    # Expected assists
                    if 'expected_assists' in group_sorted.columns:
                        col_name = f'rolling_xa_{window}'
                        roll_values = group_sorted['expected_assists'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
        
        # Calculate form score
        if 'total_points' in result_df.columns:
            for player_id, group in result_df.groupby('player_id'):
                group_sorted = group.sort_values('gameweek')
                form_scores = group_sorted['total_points'].rolling(window=5, min_periods=1).mean()
                result_df.loc[group_sorted.index, 'form_score'] = form_scores.values
        
        # Fill NaN values
        for col in result_df.columns:
            if col.startswith('rolling_') or col.endswith('_score'):
                result_df[col] = result_df[col].fillna(0)
        
        return result_df
    
    def _add_player_metadata(self, features_df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Add player position and other metadata"""
        try:
            if not self.db or features_df.empty:
                return features_df
            
            query = """
                SELECT 
                    player_id,
                    COALESCE(position, 'UNK') as position,
                    COALESCE(team_code, 0) as team_id,
                    COALESCE(web_name, 'Unknown') as web_name
                FROM players
                WHERE season = ?
            """
            
            player_data = self.db.execute_query(query, (season,))
            
            if player_data is not None and not player_data.empty:
                features_df['player_id'] = features_df['player_id'].astype(str)
                player_data['player_id'] = player_data['player_id'].astype(str)
                
                features_df = features_df.merge(
                    player_data,
                    on='player_id',
                    how='left'
                )
                print(f" Added metadata for {len(player_data)} players")
            
            return features_df
            
        except Exception as e:
            print(f" Error adding metadata: {e}")
            return features_df
    
    def _calculate_value_metrics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate value-based features"""
        if features_df.empty:
            return features_df
            
        result_df = features_df.copy()
        
        # Ensure cost column exists
        if 'now_cost' not in result_df.columns:
            result_df['now_cost'] = 5.0
        
        # Calculate points per million
        if 'total_points' in result_df.columns:
            safe_cost = result_df['now_cost'].values / 10.0
            safe_cost = np.where(safe_cost == 0, 1, safe_cost)
            
            result_df['points_per_million'] = result_df['total_points'].values / safe_cost
        
        # Calculate form per cost
        if 'form_score' in result_df.columns:
            safe_cost = result_df['now_cost'].values / 10.0
            safe_cost = np.where(safe_cost == 0, 1, safe_cost)
            
            result_df['form_per_cost'] = result_df['form_score'].values / safe_cost
        
        # Calculate ownership value
        if 'points_per_million' in result_df.columns and 'selected_by_percent' in result_df.columns:
            ownership_pct = result_df['selected_by_percent'].fillna(0).values / 100
            ownership_pct = np.where(ownership_pct == 0, 1, ownership_pct)
            
            result_df['ownership_value'] = result_df['points_per_million'].values * ownership_pct
        
        return result_df
    
    def _add_target_variable(self, features_df: pd.DataFrame, 
                           gameweek: int, season: str) -> pd.DataFrame:
        """Add target variable (points in this gameweek)"""
        try:
            if features_df.empty:
                return features_df
            
            target_query = """
                SELECT id as player_id, total_points as actual_points
                FROM player_gameweek_stats 
                WHERE gw = ? 
                   AND season = ?
            """
            
            target_data = self.db.execute_query(target_query, (gameweek, season))
            
            if target_data is not None and not target_data.empty:
                features_df['player_id'] = features_df['player_id'].astype(str)
                target_data['player_id'] = target_data['player_id'].astype(str)
                
                features_df = features_df.merge(
                    target_data,
                    on='player_id',
                    how='left'
                )
                print(f" Added target variable for {len(target_data)} players")
            
            return features_df
            
        except Exception as e:
            print(f" Error adding target variable: {e}")
            return features_df
    
    def _add_match_context(self, features_df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Add match context from matches table"""
        try:
            if not self.db or features_df.empty:
                return features_df
            
            query = """
                SELECT 
                    gameweek,
                    home_team,
                    away_team,
                    home_score,
                    away_score,
                    home_expected_goals_xg as home_xg,
                    away_expected_goals_xg as away_xg,
                    home_possession,
                    away_possession,
                    home_total_shots,
                    away_total_shots
                FROM matches
                WHERE season = ?
                ORDER BY gameweek
            """
            
            match_data = self.db.execute_query(query, (season,))
            
            if match_data is None or match_data.empty:
                print(" No match data available")
                return features_df
            
            print(f" Loaded {len(match_data)} match records for context")
            return features_df
            
        except Exception as e:
            print(f" Error adding match context: {e}")
            return features_df
    
    def _add_team_strength_metrics_fixed(self, features_df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Add team strength metrics"""
        try:
            if not self.db or features_df.empty or 'team_id' not in features_df.columns:
                print(" No team data available")
                return features_df
            
            query = """
                SELECT 
                    code as team_id,
                    name,
                    strength,
                    strength_overall_home,
                    strength_overall_away,
                    strength_attack_home,
                    strength_attack_away,
                    strength_defence_home,
                    strength_defence_away,
                    elo
                FROM teams
                WHERE season = ?
            """
            
            team_data = self.db.execute_query(query, (season,))
            
            if team_data is None or team_data.empty:
                print(" No team strength data found")
                return features_df
            
            result_df = features_df.copy()
            result_df['team_id'] = result_df['team_id'].astype(str)
            team_data['team_id'] = team_data['team_id'].astype(str)
            
            result_df = result_df.merge(
                team_data,
                on='team_id',
                how='left'
            )
            
            # Fill NaN values
            team_strength_cols = ['strength', 'elo', 'strength_overall_home', 
                                 'strength_overall_away', 'strength_attack_home',
                                 'strength_attack_away', 'strength_defence_home',
                                 'strength_defence_away']
            
            for col in team_strength_cols:
                if col in result_df.columns:
                    if col == 'strength':
                        result_df[col] = result_df[col].fillna(3.0)
                    elif col == 'elo':
                        result_df[col] = result_df[col].fillna(1500.0)
                    else:
                        result_df[col] = result_df[col].fillna(result_df[col].median() if not result_df[col].isnull().all() else 3.0)
            
            # Calculate composite metrics
            if 'strength' in result_df.columns and 'elo' in result_df.columns:
                result_df['elo_normalized'] = (result_df['elo'] - 1000) / 200
                result_df['team_strength_composite'] = (
                    result_df['strength'] * 0.7 + result_df['elo_normalized'] * 0.3
                )
            
            print(f" Added team strength metrics for {len(team_data)} teams")
            return result_df
            
        except Exception as e:
            print(f" Error adding team strength metrics: {e}")
            return features_df
    
    def _add_xg_based_metrics_fixed(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add expected goals (xG) based advanced metrics"""
        if features_df.empty:
            return features_df
        
        result_df = features_df.copy()
        xg_metrics_added = 0
        
        # 1. xG Performance vs Actual
        if 'expected_goals' in result_df.columns and 'goals_scored' in result_df.columns:
            result_df['xg_overperformance'] = result_df['goals_scored'] - result_df['expected_goals']
            result_df['xg_efficiency'] = result_df.apply(
                lambda x: x['goals_scored'] / x['expected_goals'] if x['expected_goals'] > 0 else 0,
                axis=1
            )
            xg_metrics_added += 2
        
        # 2. xA Performance vs Actual
        if 'expected_assists' in result_df.columns and 'assists' in result_df.columns:
            result_df['xa_overperformance'] = result_df['assists'] - result_df['expected_assists']
            result_df['xa_efficiency'] = result_df.apply(
                lambda x: x['assists'] / x['expected_assists'] if x['expected_assists'] > 0 else 0,
                axis=1
            )
            xg_metrics_added += 2
        
        # 3. xGI (Expected Goal Involvement) based metrics
        if 'expected_goal_involvements' in result_df.columns:
            if 'goals_scored' in result_df.columns and 'assists' in result_df.columns:
                result_df['actual_goal_involvements'] = result_df['goals_scored'] + result_df['assists']
                result_df['xgi_overperformance'] = result_df['actual_goal_involvements'] - result_df['expected_goal_involvements']
                result_df['xgi_efficiency'] = result_df.apply(
                    lambda x: x['actual_goal_involvements'] / x['expected_goal_involvements'] 
                    if x['expected_goal_involvements'] > 0 else 0,
                    axis=1
                )
                xg_metrics_added += 3
        
        # 4. xG per minute
        if 'expected_goals' in result_df.columns and 'minutes' in result_df.columns:
            result_df['xg_per_minute'] = result_df.apply(
                lambda x: x['expected_goals'] / x['minutes'] if x['minutes'] > 0 else 0,
                axis=1
            )
            xg_metrics_added += 1
        
        print(f" Added {xg_metrics_added} xG-based metrics")
        return result_df
    
    def _add_fixture_difficulty_fixed(self, features_df: pd.DataFrame, gameweek: int, season: str) -> pd.DataFrame:
        """Add fixture difficulty using matches data"""
        try:
            if not self.db or features_df.empty or 'team_id' not in features_df.columns:
                print(" No team data available for FDR calculation")
                return features_df
            
            # Get upcoming fixtures
            query = """
                SELECT 
                    gameweek,
                    home_team as team_id,
                    away_team as opponent_id,
                    1 as is_home
                FROM matches
                WHERE season = ? AND gameweek >= ?
                
                UNION ALL
                
                SELECT 
                    gameweek,
                    away_team as team_id,
                    home_team as opponent_id,
                    0 as is_home
                FROM matches
                WHERE season = ? AND gameweek >= ?
                
                ORDER BY gameweek, team_id
            """
            
            fixture_data = self.db.execute_query(query, (season, gameweek, season, gameweek))
            
            if fixture_data is None or fixture_data.empty:
                print(" No fixture data available")
                return features_df
            
            result_df = features_df.copy()
            
            # Get team Elo ratings
            team_elo_query = """
                SELECT code as team_id, elo
                FROM teams
                WHERE season = ?
            """
            team_elos = self.db.execute_query(team_elo_query, (season,))
            
            if team_elos is None or team_elos.empty:
                print(" No team Elo data available")
                return result_df
            
            # Create Elo lookup dictionary
            team_elo_dict = {}
            for _, row in team_elos.iterrows():
                team_elo_dict[str(row['team_id'])] = float(row['elo'])
            
            # Calculate FDR for each player
            fdr_added_count = 0
            
            for idx, row in result_df.iterrows():
                team_id = str(row['team_id']) if 'team_id' in row and pd.notna(row['team_id']) else None
                position = str(row['position']) if 'position' in row and pd.notna(row['position']) else 'UNK'
                
                if team_id and team_id in team_elo_dict:
                    # Get next fixture for this team
                    team_fixtures = fixture_data[fixture_data['team_id'] == float(team_id)]
                    
                    if not team_fixtures.empty:
                        next_fixture = team_fixtures.iloc[0]
                        opponent_id = str(next_fixture['opponent_id'])
                        is_home = bool(next_fixture['is_home'])
                        
                        if opponent_id in team_elo_dict:
                            team_elo = team_elo_dict[team_id]
                            opponent_elo = team_elo_dict[opponent_id]
                            
                            # Base FDR calculation
                            elo_difference = opponent_elo - team_elo
                            home_advantage_factor = -0.2 if is_home else 0.2
                            
                            base_fdr = 3.0 + (elo_difference / 200.0) + home_advantage_factor
                            
                            # Position-specific adjustments
                            if position == 'GKP':
                                position_factor = 1.1 if elo_difference > 0 else 0.9
                            elif position == 'DEF':
                                position_factor = 1.05 if elo_difference > 0 else 0.95
                            elif position == 'MID':
                                position_factor = 1.0
                            elif position == 'FWD':
                                position_factor = 0.95 if elo_difference > 0 else 1.05
                            else:
                                position_factor = 1.0
                            
                            # Apply position adjustment
                            position_fdr = base_fdr * position_factor
                            
                            # Clamp to 1-5 range
                            position_fdr = max(1.0, min(5.0, position_fdr))
                            base_fdr = max(1.0, min(5.0, base_fdr))
                            
                            result_df.loc[idx, 'position_fdr'] = position_fdr
                            result_df.loc[idx, 'base_fdr'] = base_fdr
                            result_df.loc[idx, 'position_fdr_adjustment'] = position_factor
                            result_df.loc[idx, 'is_home_next'] = is_home
                            
                            fdr_added_count += 1
            
            if fdr_added_count > 0:
                print(f" Added fixture difficulty metrics for {fdr_added_count} players")
            
            return result_df
            
        except Exception as e:
            print(f" Error adding fixture difficulty: {e}")
            return features_df
        
    def generate_comprehensive_features(self, gameweek: int, season: str = "2025-2026") -> pd.DataFrame:
        """
        Generate COMPREHENSIVE features using all enhanced feature engineering
        
        Args:
            gameweek: Gameweek to generate features for
            season: Season string
            
        Returns:
            DataFrame with comprehensive features
        """
        print(f"\n{'='*60}")
        print(f"Generating COMPREHENSIVE features for Gameweek {gameweek}, {season}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Get player history
            print("1. Loading player history...")
            player_history = self._get_player_history_complete(gameweek, season)
            
            if player_history.empty:
                print(" No player history found.")
                return pd.DataFrame()
            
            print(f" Loaded {len(player_history)} player-gameweek records")
            
            # Step 2: Get player metadata
            print("2. Loading player metadata...")
            metadata_query = """
                SELECT 
                    player_id,
                    COALESCE(position, 'UNK') as position,
                    COALESCE(team_code, 0) as team_id,
                    COALESCE(web_name, 'Unknown') as web_name
                FROM players
                WHERE season = ?
            """
            
            player_data = self.db.execute_query(metadata_query, (season,))
            
            if player_data is None or player_data.empty:
                print(" No player metadata found.")
                return pd.DataFrame()
            
            print(f" Loaded metadata for {len(player_data)} players")
            
            # Step 3: Create comprehensive features using FeatureBuilder
            print("3. Creating comprehensive features...")
            from feature_engineering.feature_builder import FeatureBuilder
            
            feature_builder = FeatureBuilder(self.db)
            features_df = feature_builder.create_comprehensive_features(
                player_history=player_history,
                player_info=player_data,
                gameweek=gameweek,
                season=season
            )
            
            if features_df.empty:
                print(" No features generated.")
                return pd.DataFrame()
            
            print(f" Generated {len(features_df)} records with {len(features_df.columns)} features")
            
            # Step 4: Add team strength metrics
            print("4. Adding team strength metrics...")
            features_df = self._add_team_strength_metrics_fixed(features_df, season)
            
            # Step 5: Add fixture difficulty
            print("5. Adding fixture difficulty...")
            features_df = self._add_fixture_difficulty_fixed(features_df, gameweek, season)
            
            # Step 6: Add match context
            print("6. Adding match context...")
            features_df = self._add_match_context(features_df, season)
            
            # Step 7: Add target variable
            print("7. Adding target variable...")
            features_df = self._add_target_variable(features_df, gameweek, season)
            
            # Step 8: Clean and finalize
            print("8. Cleaning and finalizing features...")
            features_df = self._clean_features_enhanced(features_df)
            
            print(f"\n✓ COMPREHENSIVE feature generation complete!")
            print(f" Final dataset: {len(features_df)} records, {len(features_df.columns)} features")
            
            # Show feature categories
            self._print_comprehensive_feature_categories(features_df)
            
            return features_df
            
        except Exception as e:
            print(f"✗ Error generating comprehensive features: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _print_comprehensive_feature_categories(self, features_df: pd.DataFrame):
        """Print comprehensive feature categories summary"""
        if features_df.empty:
            return
        
        categories = {
            'Status & Availability': [col for col in features_df.columns 
                                     if any(keyword in col for keyword in 
                                           ['status', 'playing', 'chance', 'probability', 'available'])],
            'Form & Performance': [col for col in features_df.columns 
                                  if any(keyword in col for keyword in 
                                        ['form', 'score', 'points_', 'bps', 'consistency'])],
            'Ownership & Market': [col for col in features_df.columns 
                                  if any(keyword in col for keyword in 
                                        ['selected', 'ownership', 'transfer', 'cost_change', 'value'])],
            'Expected Metrics (xG/xA/xGI)': [col for col in features_df.columns 
                                            if any(keyword in col for keyword in 
                                                  ['expected', 'xg', 'xa', 'xgi', 'xgc', 'ep_'])],
            'Rolling Performance': [col for col in features_df.columns 
                                   if 'rolling_' in col],
            'Team Strength': [col for col in features_df.columns 
                             if any(keyword in col for keyword in 
                                   ['strength', 'elo', 'team_'])],
            'Fixture Difficulty': [col for col in features_df.columns 
                                 if 'fdr' in col.lower()],
            'ICT Index': [col for col in features_df.columns 
                         if col in ['influence', 'creativity', 'threat', 'ict_index']],
            'Position-Specific': [col for col in features_df.columns 
                                if any(keyword in col for keyword in 
                                      ['saves_per', 'conceded_per', 'def_actions', 'tackles_per', 
                                       'xa_per', 'xg_per', 'clean_sheet_prob', 'goal_scoring_prob'])],
            'Value Metrics': [col for col in features_df.columns 
                             if any(keyword in col for keyword in 
                                   ['cost', 'million', 'value', 'per_cost', 'for_money'])],
            'Basic Stats': ['minutes', 'total_points', 'goals_scored', 'assists', 
                           'clean_sheets', 'goals_conceded', 'bonus', 'saves', 'tackles']
        }
        
        print("\nCOMPREHENSIVE Feature Categories Summary:")
        for category, features in categories.items():
            existing_features = [f for f in features if f in features_df.columns]
            if existing_features:
                print(f" {category}: {len(existing_features)} features")
                # Print top 5 features in each category
                if existing_features and len(existing_features) > 0:
                    print(f"   Top: {', '.join(existing_features[:5])}")

class FeaturePipeline(FixedFeaturePipeline):
    """Main feature pipeline class (backward compatible)"""
    pass