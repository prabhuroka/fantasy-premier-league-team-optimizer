"""
Core feature building logic with error handling
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureBuilder:
    def __init__(self, db=None):
        """
        Initialize FeatureBuilder with database connection.
        
        Args:
            db: FPLDatabase instance for data access
        """
        self.db = db
        self.position_mapping = {
            'GKP': 'Goalkeeper',
            'DEF': 'Defender', 
            'MID': 'Midfielder',
            'FWD': 'Forward'
        }
        
    def calculate_rolling_features(self, player_history: pd.DataFrame, 
                                  windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """
        Calculate rolling averages for key metrics.
        
        Args:
            player_history: DataFrame with player gameweek stats
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        if player_history.empty:
            return player_history
            
        # Sort by gameweek
        player_history = player_history.sort_values(['player_id', 'gameweek'])
        
        # Initialize result DataFrame
        result_df = player_history.copy()
        
        # Group by player to calculate rolling stats
        for player_id, group in result_df.groupby('player_id'):
            group_sorted = group.sort_values('gameweek')
            
            # Calculate rolling features for each window
            for window in windows:
                if window <= len(group_sorted):
                    # Points rolling average
                    if 'total_points' in group_sorted.columns:
                        col_name = f'rolling_points_{window}'
                        roll_values = group_sorted['total_points'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                    
                    # Minutes rolling average
                    if 'minutes' in group_sorted.columns:
                        col_name = f'rolling_minutes_{window}'
                        roll_values = group_sorted['minutes'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                        
                        # Minutes consistency
                        col_name = f'minutes_consistency_{window}'
                        minutes_bool = (group_sorted['minutes'] > 60).astype(int)
                        consistency = minutes_bool.rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = consistency.values
                    
                    # Goalkeeper-specific rolling features
                    if 'saves' in group_sorted.columns:
                        col_name = f'rolling_saves_{window}'
                        roll_values = group_sorted['saves'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                    
                    if 'goals_conceded' in group_sorted.columns:
                        col_name = f'rolling_conceded_{window}'
                        roll_values = group_sorted['goals_conceded'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                    
                    # Expected goals rolling average
                    if 'expected_goals' in group_sorted.columns:
                        col_name = f'rolling_xg_{window}'
                        roll_values = group_sorted['expected_goals'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
                    
                    # Expected assists rolling average
                    if 'expected_assists' in group_sorted.columns:
                        col_name = f'rolling_xa_{window}'
                        roll_values = group_sorted['expected_assists'].rolling(window=window, min_periods=1).mean()
                        result_df.loc[group_sorted.index, col_name] = roll_values.values
        
        # Fill NaN values with appropriate defaults
        for col in result_df.columns:
            if col.startswith('rolling_'):
                result_df[col] = result_df[col].fillna(0)
        
        return result_df
    
    def calculate_form_metrics(self, player_data: pd.DataFrame, 
                              recent_gws: int = 5) -> pd.DataFrame:
        """
        Calculate form indicators for players.
        
        Args:
            player_data: DataFrame with player gameweek stats (must have rolling features)
            recent_gws: Number of recent gameweeks to consider
            
        Returns:
            DataFrame with form metrics added
        """
        if player_data.empty:
            return player_data
            
        result_df = player_data.copy()
        
        # Ensure data is sorted by gameweek
        result_df = result_df.sort_values(['player_id', 'gameweek'])
        
        # Calculate form score (exponential weighted average)
        for player_id, group in result_df.groupby('player_id'):
            group_sorted = group.sort_values('gameweek')
            
            if 'total_points' in group_sorted.columns:
                # Calculate weighted average (recent games weighted more heavily)
                weights = np.exp(np.linspace(0, 1, min(recent_gws, len(group_sorted))))
                weights = weights / weights.sum()
                
                # Apply weighted average to last 'recent_gws' games
                for idx in group_sorted.index:
                    player_gw = group_sorted.loc[idx, 'gameweek']
                    recent_games = group_sorted[group_sorted['gameweek'] < player_gw].tail(recent_gws)
                    
                    if len(recent_games) > 0:
                        # Use appropriate number of weights
                        use_weights = weights[-len(recent_games):] if len(recent_games) <= len(weights) else weights
                        use_weights = use_weights / use_weights.sum()
                        
                        form_score = np.average(recent_games['total_points'].values, weights=use_weights)
                        result_df.loc[idx, 'form_score'] = form_score
                        
                        # Calculate consistency (1 - coefficient of variation)
                        if len(recent_games) >= 2:
                            std_dev = recent_games['total_points'].std()
                            mean = recent_games['total_points'].mean()
                            if mean > 0:
                                cv = std_dev / mean
                                result_df.loc[idx, 'consistency_score'] = max(0, 1 - cv)
                            else:
                                result_df.loc[idx, 'consistency_score'] = 0
                        else:
                            result_df.loc[idx, 'consistency_score'] = 0.5
        
        # Fill NaN values
        form_cols = ['form_score', 'consistency_score']
        for col in form_cols:
            if col in result_df.columns:
                if col == 'consistency_score':
                    result_df[col] = result_df[col].fillna(0.5)
                else:
                    result_df[col] = result_df[col].fillna(0)
        
        return result_df
    
    def add_player_status_features(self, features_df: pd.DataFrame, 
                                   gameweek: int, 
                                   season: str = "2025-2026") -> pd.DataFrame:
        """
        FIXED: Add player status features with error handling and duplicate handling
        
        Args:
            features_df: Base features DataFrame
            gameweek: Current gameweek
            season: Season string
            
        Returns:
            DataFrame with status features added
        """
        if features_df.empty or not self.db:
            return features_df
        
        try:
            # Get player status data for current gameweek from player_stats table
            query = """
                SELECT 
                    id as player_id,
                    status,
                    chance_of_playing_next_round,
                    chance_of_playing_this_round,
                    form as form_current,
                    points_per_game,
                    starts,
                    influence as influence_current,
                    creativity as creativity_current,
                    threat as threat_current,
                    ict_index as ict_index_current,
                    bps as bps_current,
                    transfers_in,
                    transfers_out,
                    value_form,
                    value_season,
                    ep_next,
                    ep_this,
                    now_cost,
                    cost_change_event,
                    defensive_contribution
                FROM player_stats 
                WHERE gw = ? 
                  AND season = ?
            """
            
            status_data = self.db.execute_query(query, (gameweek, season))
            
            if status_data is None or status_data.empty:
                print(f"  ⚠ No player status data found for GW{gameweek}")
                return features_df
            
            # Check for duplicate player_ids in status_data
            duplicate_players = status_data[status_data.duplicated('player_id', keep=False)]
            if not duplicate_players.empty:
                print(f"  ⚠ Found {len(duplicate_players)} duplicate player records in status data")
                # Keep the first record for each duplicate
                status_data = status_data.drop_duplicates(subset='player_id', keep='first')
            
            # Standardize player_id type
            features_df['player_id'] = features_df['player_id'].astype(str)
            status_data['player_id'] = status_data['player_id'].astype(str)
            
            # Create a copy to avoid modifying original
            result_df = features_df.copy()
            
            # Merge using pandas merge instead of dictionary to handle duplicates
            status_cols = [col for col in status_data.columns if col != 'player_id']
            
            for col in status_cols:
                # Create mapping from player_id to value for this column
                col_mapping = status_data.set_index('player_id')[col].to_dict()
                
                # Map the values to result_df
                result_df[f'{col}'] = result_df['player_id'].map(col_mapping)
            
            # Process status features
            # 1. Convert status to numeric (critical for predictions)
            status_mapping = {'a': 1, 'i': 0, 'u': 0, 'd': 0.5}
            result_df['status_numeric'] = result_df['status'].map(status_mapping).fillna(0)
            
            # 2. Process playing chances (scale 0-1)
            result_df['playing_chance_next'] = result_df['chance_of_playing_next_round'].fillna(0) / 100.0
            result_df['playing_chance_this'] = result_df['chance_of_playing_this_round'].fillna(0) / 100.0
            
            # 3. Create composite playing probability
            result_df['playing_probability'] = (
                result_df['playing_chance_next'].fillna(0.5) * 0.6 + 
                result_df['playing_chance_this'].fillna(0.5) * 0.4
            )
            
            # 4. Form feature with playing probability adjustment (CRITICAL)
            result_df['form_adjusted'] = result_df['form_current'].fillna(0) * result_df['playing_probability']
            
            # 5. Points per game adjusted
            result_df['ppg_adjusted'] = result_df['points_per_game'].fillna(0) * result_df['playing_probability']
            
            # 6. Expected points features (with data leakage protection)
            # Only use ep_next (future) not ep_this (current gameweek)
            result_df['ep_next'] = result_df['ep_next'].fillna(0)
            result_df['ep_total'] = result_df['ep_next']  # Only use future expected points
            
            # 7. ICT Index (Influence, Creativity, Threat) - current status
            result_df['influence_current'] = result_df['influence_current'].fillna(0)
            result_df['creativity_current'] = result_df['creativity_current'].fillna(0)
            result_df['threat_current'] = result_df['threat_current'].fillna(0)
            result_df['ict_index_current'] = result_df['ict_index_current'].fillna(0)
            result_df['bps_current'] = result_df['bps_current'].fillna(0)
            
            # Remove any _status columns that might have been created
            status_cols_to_remove = [col for col in result_df.columns if col.endswith('_status')]
            if status_cols_to_remove:
                result_df = result_df.drop(columns=status_cols_to_remove)
            
            print(f"  Added player status features for {len(status_data)} players")
            
            return result_df
            
        except Exception as e:
            print(f"  ⚠ Error adding player status features: {e}")
            # Return features_df with basic playing_probability column
            features_df['playing_probability'] = 0.5
            features_df['status_numeric'] = 1.0
            features_df['form_adjusted'] = features_df.get('form_score', 0)
            return features_df
    
    def add_position_specific_status_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Add position-specific features with column existence checks
        
        Args:
            features_df: Features DataFrame with status data
            
        Returns:
            DataFrame with position-specific status features
        """
        if features_df.empty or 'position' not in features_df.columns:
            return features_df
        
        result_df = features_df.copy()
        
        # Ensure playing_probability exists
        if 'playing_probability' not in result_df.columns:
            result_df['playing_probability'] = 0.5
        
        # Goalkeeper specific status features - ENHANCED
        goalkeepers = result_df[result_df['position'] == 'Goalkeeper'].copy()
        if not goalkeepers.empty:
            # Expected goals conceded per 90
            if 'expected_goals_conceded' in goalkeepers.columns and 'minutes' in goalkeepers.columns:
                result_df.loc[goalkeepers.index, 'xgc_per_90'] = goalkeepers.apply(
                    lambda row: row['expected_goals_conceded'] * 90 / row['minutes'] if row['minutes'] > 0 else 0,
                    axis=1
                )
                result_df.loc[goalkeepers.index, 'xgc_per_90_adj'] = (
                    result_df.loc[goalkeepers.index, 'xgc_per_90'].fillna(0) * 
                    result_df.loc[goalkeepers.index, 'playing_probability'].fillna(0.5)
                )
            
            # Clean sheet probability based on expected goals conceded
            if 'expected_goals_conceded' in goalkeepers.columns:
                result_df.loc[goalkeepers.index, 'clean_sheet_prob'] = np.exp(-goalkeepers['expected_goals_conceded'].fillna(0) / 2)
                result_df.loc[goalkeepers.index, 'clean_sheet_prob_adj'] = (
                    result_df.loc[goalkeepers.index, 'clean_sheet_prob'].fillna(0) * 
                    result_df.loc[goalkeepers.index, 'playing_probability'].fillna(0.5)
                )
        
        # Defender specific status features
        defenders = result_df[result_df['position'] == 'Defender'].copy()
        if not defenders.empty:
            # Clean sheet probability based on expected goals conceded
            if 'expected_goals_conceded' in defenders.columns:
                result_df.loc[defenders.index, 'clean_sheet_prob'] = np.exp(-defenders['expected_goals_conceded'].fillna(0) / 3)
                result_df.loc[defenders.index, 'clean_sheet_prob_adj'] = (
                    result_df.loc[defenders.index, 'clean_sheet_prob'].fillna(0) * 
                    result_df.loc[defenders.index, 'playing_probability'].fillna(0.5)
                )
            
            # Defensive actions per 90
            if 'clearances_blocks_interceptions' in defenders.columns and 'minutes' in defenders.columns:
                result_df.loc[defenders.index, 'def_actions_per_90'] = defenders.apply(
                    lambda row: row['clearances_blocks_interceptions'] * 90 / row['minutes'] if row['minutes'] > 0 else 0,
                    axis=1
                )
        
        # Midfielder specific status features
        midfielders = result_df[result_df['position'] == 'Midfielder'].copy()
        if not midfielders.empty:
            # Creativity and threat features
            if 'creativity_current' in midfielders.columns and 'threat_current' in midfielders.columns:
                # Handle division by zero
                creativity_filled = midfielders['creativity_current'].fillna(1)
                threat_filled = midfielders['threat_current'].fillna(1)
                result_df.loc[midfielders.index, 'creativity_threat_ratio'] = (
                    creativity_filled / threat_filled.replace(0, 1)
                )
                result_df.loc[midfielders.index, 'ct_index'] = (
                    midfielders['creativity_current'].fillna(0) + midfielders['threat_current'].fillna(0)
                )
        
        # Forward specific status features
        forwards = result_df[result_df['position'] == 'Forward'].copy()
        if not forwards.empty:
            # Goal scoring probability based on xG
            if 'expected_goals' in forwards.columns:
                result_df.loc[forwards.index, 'goal_scoring_probability'] = 1 - np.exp(-forwards['expected_goals'].fillna(0))
                result_df.loc[forwards.index, 'goal_scoring_probability_adj'] = (
                    result_df.loc[forwards.index, 'goal_scoring_probability'].fillna(0) * 
                    result_df.loc[forwards.index, 'playing_probability'].fillna(0.5)
                )
        
        return result_df
    
    def calculate_derived_status_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from player status data
        
        Args:
            features_df: Features DataFrame with status data
            
        Returns:
            DataFrame with derived features
        """
        if features_df.empty:
            return features_df
        
        result_df = features_df.copy()
        
        # Ensure playing_probability exists
        if 'playing_probability' not in result_df.columns:
            result_df['playing_probability'] = 0.5
        
        # 1. Expected points adjusted by availability
        if 'ep_total' in result_df.columns:
            result_df['ep_total_adj'] = result_df['ep_total'] * result_df['playing_probability']
        
        # 2. Consistency score based on starts and minutes
        if 'starts' in result_df.columns and 'minutes_consistency_3' in result_df.columns:
            result_df['overall_consistency'] = (
                result_df['starts'].fillna(0) * 0.6 + 
                result_df['minutes_consistency_3'].fillna(0) * 0.4
            )
        
        # 3. Availability-adjusted rolling features
        rolling_cols = [col for col in result_df.columns if 'rolling_' in col]
        for col in rolling_cols:
            if col + '_adj' not in result_df.columns:
                adj_col = col + '_adj'
                result_df[adj_col] = (
                    result_df[col].fillna(0) * 
                    result_df['playing_probability'].fillna(0.5)
                )
        
        return result_df
    
    def create_comprehensive_features(self, player_history: pd.DataFrame, 
                                     player_info: pd.DataFrame = None,
                                     gameweek: int = None,
                                     season: str = "2025-2026") -> pd.DataFrame:
        """
        FIXED: Create comprehensive feature set with proper error handling
        
        Args:
            player_history: Gameweek-level player stats
            player_info: Player metadata (optional)
            gameweek: Current gameweek (required for status features)
            season: Season string
            
        Returns:
            Comprehensive feature DataFrame
        """
        if player_history.empty:
            return pd.DataFrame()
        
        print(f"Creating comprehensive features for {len(player_history)} records...")
        
        try:
            # Step 1: Calculate rolling features
            print("  Step 1: Calculating rolling features...")
            features_df = self.calculate_rolling_features(player_history)
            
            # Step 2: Calculate form metrics
            print("  Step 2: Calculating form metrics...")
            features_df = self.calculate_form_metrics(features_df)
            
            # Step 3: Merge player info if provided
            if player_info is not None and not player_info.empty:
                print("  Step 3: Merging player information...")
                
                if 'player_id' in features_df.columns and 'player_id' in player_info.columns:
                    features_df['player_id'] = features_df['player_id'].astype(str)
                    player_info['player_id'] = player_info['player_id'].astype(str)
                    
                    features_df = features_df.merge(
                        player_info,
                        on='player_id',
                        how='left'
                    )
            
            # Step 4: Add player status features (if gameweek provided)
            if gameweek is not None and self.db is not None:
                print("  Step 4: Adding player status and availability features...")
                features_df = self.add_player_status_features(features_df, gameweek, season)
                
                print("  Step 5: Adding position-specific status features...")
                features_df = self.add_position_specific_status_features(features_df)
                
                print("  Step 6: Calculating derived status features...")
                features_df = self.calculate_derived_status_features(features_df)
            else:
                print("  ⚠ Skipping status features (no gameweek or database)")
                # Add basic playing_probability if not present
                if 'playing_probability' not in features_df.columns:
                    features_df['playing_probability'] = 0.5
                if 'status_numeric' not in features_df.columns:
                    features_df['status_numeric'] = 1.0
                if 'form_adjusted' not in features_df.columns:
                    features_df['form_adjusted'] = features_df.get('form_score', 0)
            
            # Step 7: Calculate position-specific features (CRITICAL FIX)
            print("  Step 7: Calculating position-specific features...")
            features_df = self._add_all_position_features_enhanced(features_df)
            
            # Step 8: Add derived features
            print("  Step 8: Adding derived features...")
            
            # Minutes played categories
            if 'minutes' in features_df.columns:
                features_df['played_90'] = (features_df['minutes'] >= 90).astype(int)
                features_df['played_60'] = (features_df['minutes'] >= 60).astype(int)
                features_df['played_30'] = (features_df['minutes'] >= 30).astype(int)
            
            # Bonus point involvement
            if 'bonus' in features_df.columns:
                features_df['got_bonus'] = (features_df['bonus'] > 0).astype(int)
            
            # Step 9: Add xG-based metrics
            print("  Step 9: Adding xG-based metrics...")
            features_df = self._add_xg_based_metrics(features_df)
            
            # Step 10: Ensure all critical features are present
            print("  Step 10: Ensuring critical features...")
            features_df = self._ensure_all_critical_features(features_df)
            
            # Step 11: Remove any remaining contaminated features
            print("  Step 11: Cleaning feature names...")
            features_df = self._clean_contaminated_features(features_df)
            
            print(f"  ✓ Comprehensive feature engineering complete. Generated {len(features_df.columns)} features.")
            
            return features_df
            
        except Exception as e:
            print(f"  ✗ Error in create_comprehensive_features: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _add_all_position_features_enhanced(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add ALL position-specific features comprehensively - FIXED VERSION"""
        if features_df.empty or 'position' not in features_df.columns:
            return features_df
    
        result_df = features_df.copy()
        
        # Goalkeeper specific features - ENHANCED
        goalkeepers = result_df[result_df['position'] == 'Goalkeeper'].copy()
        if not goalkeepers.empty:
            print(f"    Adding enhanced goalkeeper features for {len(goalkeepers)} players")
            
            # Saves per 90 (CRITICAL)
            if 'saves' in goalkeepers.columns and 'minutes' in goalkeepers.columns:
                result_df.loc[goalkeepers.index, 'saves_per_90'] = goalkeepers.apply(
                    lambda row: (row['saves'] * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
            
            # Goals conceded per 90 (CRITICAL)
            if 'goals_conceded' in goalkeepers.columns and 'minutes' in goalkeepers.columns:
                result_df.loc[goalkeepers.index, 'conceded_per_90'] = goalkeepers.apply(
                    lambda row: (row['goals_conceded'] * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
            
            # Save percentage
            if 'saves' in goalkeepers.columns and 'goals_conceded' in goalkeepers.columns:
                shots_faced = goalkeepers['saves'] + goalkeepers['goals_conceded']
                result_df.loc[goalkeepers.index, 'save_percentage'] = (
                    goalkeepers['saves'] / shots_faced.replace(0, 1)
                ).fillna(0.7)  # Default 70%
            
            # Clean sheet probability
            if 'expected_goals_conceded' in goalkeepers.columns:
                result_df.loc[goalkeepers.index, 'gk_clean_sheet_prob'] = np.exp(-goalkeepers['expected_goals_conceded'].fillna(0) / 2)
            
            # Penalty save rate
            if 'penalties_saved' in goalkeepers.columns:
                result_df.loc[goalkeepers.index, 'penalty_save_rate'] = goalkeepers['penalties_saved'].fillna(0) / 10
            
            # Bonus points per 90
            if 'bonus' in goalkeepers.columns and 'minutes' in goalkeepers.columns:
                result_df.loc[goalkeepers.index, 'gk_bonus_per_90'] = goalkeepers.apply(
                    lambda row: (row['bonus'] * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
        
        # Defender specific features
        defenders = result_df[result_df['position'] == 'Defender'].copy()
        if not defenders.empty:
            print(f"    Adding defender features for {len(defenders)} players")
            
            # Defensive actions per 90
            if 'clearances_blocks_interceptions' in defenders.columns and 'minutes' in defenders.columns:
                result_df.loc[defenders.index, 'def_actions_per_90'] = defenders.apply(
                    lambda row: (row['clearances_blocks_interceptions'] * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
            
            # Clean sheet probability
            if 'expected_goals_conceded' in defenders.columns:
                result_df.loc[defenders.index, 'def_clean_sheet_prob'] = np.exp(-defenders['expected_goals_conceded'].fillna(0) / 3)
            
            # Tackles per 90
            if 'tackles' in defenders.columns and 'minutes' in defenders.columns:
                result_df.loc[defenders.index, 'tackles_per_90'] = defenders.apply(
                    lambda row: (row['tackles'] * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
    
        # Midfielder specific features
        midfielders = result_df[result_df['position'] == 'Midfielder'].copy()
        if not midfielders.empty:
            print(f"    Adding midfielder features for {len(midfielders)} players")
            
            # Expected assists per 90
            if 'expected_assists' in midfielders.columns and 'minutes' in midfielders.columns:
                result_df.loc[midfielders.index, 'xa_per_90'] = midfielders.apply(
                    lambda row: (row['expected_assists'] * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
            
            # Goal involvement per 90
            if 'goals_scored' in midfielders.columns and 'assists' in midfielders.columns and 'minutes' in midfielders.columns:
                result_df.loc[midfielders.index, 'goal_involvement_per_90'] = midfielders.apply(
                    lambda row: ((row['goals_scored'] + row['assists']) * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
            
            # Creativity to threat ratio
            if 'creativity_current' in midfielders.columns and 'threat_current' in midfielders.columns:
                result_df.loc[midfielders.index, 'creativity_threat_ratio'] = (
                    midfielders['creativity_current'].fillna(1) / midfielders['threat_current'].replace(0, 1)
                )
        
        # Forward specific features
        forwards = result_df[result_df['position'] == 'Forward'].copy()
        if not forwards.empty:
            print(f"    Adding forward features for {len(forwards)} players")
            
            # Expected goals per 90 (CRITICAL)
            if 'expected_goals' in forwards.columns and 'minutes' in forwards.columns:
                result_df.loc[forwards.index, 'xg_per_90'] = forwards.apply(
                    lambda row: (row['expected_goals'] * 90 / row['minutes']) if row['minutes'] > 0 else 0,
                    axis=1
                )
            
            # Goal efficiency
            if 'goals_scored' in forwards.columns and 'expected_goals' in forwards.columns:
                result_df.loc[forwards.index, 'goal_efficiency'] = (
                    forwards['goals_scored'].fillna(0) / forwards['expected_goals'].replace(0, 1)
                ).fillna(0)
        
            # Threat per minute
            if 'threat_current' in forwards.columns and 'minutes' in forwards.columns:
                result_df.loc[forwards.index, 'threat_per_minute'] = forwards.apply(
                    lambda row: row['threat_current'] / row['minutes'] if row['minutes'] > 0 else 0,
                    axis=1
                )
    
        return result_df

    def _add_xg_based_metrics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive xG-based metrics"""
        if features_df.empty:
            return features_df
        
        result_df = features_df.copy()
        
        # 1. xG Performance vs Actual
        if 'expected_goals' in result_df.columns and 'goals_scored' in result_df.columns:
            result_df['xg_overperformance'] = result_df['goals_scored'] - result_df['expected_goals']
            result_df['xg_efficiency'] = result_df.apply(
                lambda x: x['goals_scored'] / x['expected_goals'] if x['expected_goals'] > 0 else 0,
                axis=1
            )
        
        # 2. xA Performance vs Actual
        if 'expected_assists' in result_df.columns and 'assists' in result_df.columns:
            result_df['xa_overperformance'] = result_df['assists'] - result_df['expected_assists']
            result_df['xa_efficiency'] = result_df.apply(
                lambda x: x['assists'] / x['expected_assists'] if x['expected_assists'] > 0 else 0,
                axis=1
            )
        
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
        
        return result_df

    def _ensure_all_critical_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ALL critical features are present with proper defaults"""
        result_df = features_df.copy()
        
        # Comprehensive list of critical features for RAW point prediction
        critical_features = {
            # Status & Availability (MOST IMPORTANT)
            'status_numeric': 1.0,
            'playing_probability': 0.5,
            'playing_chance_next': 0.5,
            'playing_chance_this': 0.5,
            
            # Form & Performance
            'form_adjusted': 0.0,
            'form_score': 0.0,
            'consistency_score': 0.5,
            
            # Rolling Performance
            'rolling_points_3': 0.0,
            'rolling_points_5': 0.0,
            'rolling_minutes_3': 0.0,
            
            # Position-Specific (CRITICAL FIX)
            'saves_per_90': 0.0,           # Goalkeeper
            'conceded_per_90': 0.0,        # Goalkeeper
            'clean_sheet_prob': 0.0,       # Goalkeeper/Defender
            'def_actions_per_90': 0.0,     # Defender
            'tackles_per_90': 0.0,         # Defender
            'xa_per_90': 0.0,              # Midfielder
            'xg_per_90': 0.0,              # Forward
            'goal_involvement_per_90': 0.0,# Midfielder
            
            # xG Metrics
            'xg_overperformance': 0.0,
            'xa_overperformance': 0.0,
            'xgi_overperformance': 0.0,
            
            # Play time indicators
            'played_90': 0,
            'played_60': 0,
            'played_30': 0,
            'got_bonus': 0,
            
            # Team context
            'team_strength_composite': 3.0,
        }
        
        # Add any missing critical features
        for feature, default_value in critical_features.items():
            if feature not in result_df.columns:
                result_df[feature] = default_value
    
        return result_df

    def _clean_contaminated_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove contaminated features (_status suffix, cost/value features)"""
        result_df = features_df.copy()
        
        # Remove all columns ending with _status
        status_cols = [col for col in result_df.columns if col.endswith('_status')]
        if status_cols:
            print(f"    Removing {len(status_cols)} _status columns")
            result_df = result_df.drop(columns=status_cols)
        
        # Remove cost/value related columns (for RAW point prediction)
        cost_keywords = ['cost', 'value', 'price', 'million', 'transfer', 'selected_by']
        cost_cols = []
        
        for col in result_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in cost_keywords):
                # Keep these critical performance features
                if col not in ['playing_probability', 'form_adjusted', 
                              'playing_chance_next', 'playing_chance_this']:
                    cost_cols.append(col)
        
        if cost_cols:
            print(f"    Removing {len(cost_cols)} cost/value columns")
            result_df = result_df.drop(columns=cost_cols)
        
        # Remove ep_this to prevent data leakage
        if 'ep_this' in result_df.columns:
            print(f"    Removing ep_this (data leakage risk)")
            result_df = result_df.drop(columns=['ep_this'])
        
        return result_df