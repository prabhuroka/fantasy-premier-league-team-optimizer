"""
Load and prepare enhanced training data for RAW point prediction 
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import warnings
import sqlite3
warnings.filterwarnings('ignore')

from ml_model.config import FEATURES_DIR, TARGET_COLUMN, POSITIONS, DATABASE_PATH, EXCLUDED_FEATURES_RAW


class DataLoader:
    def __init__(self):
        """Initialize DataLoader with paths to enhanced features"""
        self.features_dir = FEATURES_DIR
        self.target_column = TARGET_COLUMN
        self.db_path = DATABASE_PATH
        self.excluded_features = EXCLUDED_FEATURES_RAW
        
    def get_latest_gameweek(self, season: str = "2025-2026") -> int:
        """
        Get the latest available gameweek from database
        
        Returns:
            Latest gameweek number
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT MAX(gw) as latest_gw 
                FROM player_gameweek_stats 
                WHERE season = ?
            """
            result = pd.read_sql_query(query, conn, params=(season,))
            conn.close()
            
            if result is not None and not result.empty and result.iloc[0]['latest_gw']:
                return int(result.iloc[0]['latest_gw'])
            return 1  # Default to GW1 if no data
            
        except Exception as e:
            print(f"Error getting latest gameweek: {e}")
            return 1
    
    def get_available_gameweeks(self, season: str = "2025-2026") -> List[int]:
        """
        Get list of available gameweeks from database
        
        Returns:
            List of available gameweek numbers
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT DISTINCT gw 
                FROM player_gameweek_stats 
                WHERE season = ?
                ORDER BY gw
            """
            result = pd.read_sql_query(query, conn, params=(season,))
            conn.close()
            
            if result is not None and not result.empty:
                return sorted(result['gw'].tolist())
            return [1]  # Default to GW1 if no data
            
        except Exception as e:
            print(f"Error getting available gameweeks: {e}")
            return [1]
        
    def load_complete_training_data(self, force_regenerate: bool = False) -> pd.DataFrame:
        """
        Load or generate complete training data with ALL gameweeks
    
        Args:
            force_regenerate: Whether to force regeneration of features
            
        Returns:
            DataFrame with complete training data
        """
        complete_file = os.path.join(self.features_dir, 'complete_training_data.csv')
        
        if force_regenerate or not os.path.exists(complete_file):
            print("   Complete training data not found or regeneration requested")
            print("   Run: python feature_engineering/run_feature_engineering.py")
            raise FileNotFoundError("No training data found. Run feature engineering first.")
        
        print(f"Loading complete training data from: {complete_file}")
        data = pd.read_csv(complete_file)
        
        # Standardize position names
        if 'position' in data.columns:
            position_mapping = {
                'GKP': 'Goalkeeper',
                'Goalkeeper': 'Goalkeeper',
                'DEF': 'Defender',
                'Defender': 'Defender',
                'MID': 'Midfielder',
                'Midfielder': 'Midfielder',
                'FWD': 'Forward',
                'Forward': 'Forward'
            }
            data['position'] = data['position'].map(position_mapping).fillna(data['position'])
        
        print(f"  Loaded {len(data)} samples with {len(data.columns)} features")
        
        # Print position distribution
        if 'position' in data.columns:
            position_counts = data['position'].value_counts()
            print(f"  Position distribution: {position_counts.to_dict()}")
            
            # Check for sufficient goalkeeper data
            gk_count = position_counts.get('Goalkeeper', 0)
            if gk_count < 20:
                print(f"  ⚠ WARNING: Only {gk_count} goalkeeper samples (minimum 20 recommended)")
        
        return data
    
    def load_raw_training_data(self, exclude_value_features: bool = True) -> pd.DataFrame:
        """
        Load training data and exclude cost/value features for RAW point prediction
        
        Args:
            exclude_value_features: Whether to exclude cost/value features
            
        Returns:
            DataFrame with raw training data (no cost/value features)
        """
        # Load complete training data
        data = self.load_complete_training_data()
        
        if exclude_value_features:
            # First pass: remove explicitly excluded features
            existing_exclude = [f for f in self.excluded_features if f in data.columns]
            if existing_exclude:
                print(f"  Removing {len(existing_exclude)} explicitly excluded features")
                data = data.drop(columns=existing_exclude)
            
            # Second pass: aggressive filtering of contaminated features
            data = self.filter_contaminated_features(data)
            
            # Print remaining feature count
            feature_count = len([col for col in data.columns if col not in [
                'player_id', 'season', 'feature_generation_time', 
                'target_gameweek', 'actual_points', 'web_name', 'team_id'
            ]])
            print(f"  Remaining features for raw point prediction: {feature_count}")
            
            # Validate position-specific features are present
            self.validate_position_features(data)
        
        return data
    
    def filter_contaminated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive filtering of contaminated features (_status suffix, cost/value patterns)
        
        Args:
            data: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        result = data.copy()
        
        # Remove all columns ending with _status
        status_cols = [col for col in result.columns if col.endswith('_status')]
        if status_cols:
            print(f"  Removing {len(status_cols)} _status columns")
            result = result.drop(columns=status_cols)
        
        # Remove cost/value related columns by pattern
        cost_patterns = ['cost', 'value', 'price', 'million', 'transfer', 'selected_by']
        cost_cols = []
        
        for col in result.columns:
            col_lower = col.lower()
            # Check for cost patterns but keep critical performance features
            if any(pattern in col_lower for pattern in cost_patterns):
                # Keep these critical performance features
                if col not in ['playing_probability', 'form_adjusted', 
                              'playing_chance_next', 'playing_chance_this']:
                    cost_cols.append(col)
        
        if cost_cols:
            print(f"  Removing {len(cost_cols)} cost/value columns")
            result = result.drop(columns=cost_cols)
        
        # Remove ep_this to prevent data leakage
        if 'ep_this' in result.columns:
            print(f"  Removing ep_this (data leakage risk)")
            result = result.drop(columns=['ep_this'])
        
        return result
    
    def validate_position_features(self, data: pd.DataFrame) -> None:
        """
        Validate that position-specific features are present
        
        Args:
            data: DataFrame to validate
        """
        position_features = {
            'Goalkeeper': ['saves_per_90', 'conceded_per_90', 'clean_sheet_prob'],
            'Defender': ['def_actions_per_90', 'tackles_per_90'],
            'Midfielder': ['xa_per_90', 'goal_involvement_per_90'],
            'Forward': ['xg_per_90', 'goal_efficiency']
        }
        
        for position, required_features in position_features.items():
            pos_data = data[data['position'] == position]
            if len(pos_data) > 0:
                present_features = [f for f in required_features if f in pos_data.columns]
                missing_features = [f for f in required_features if f not in pos_data.columns]
                
                if missing_features:
                    print(f"  ⚠ {position}: Missing {len(missing_features)} position-specific features")
                else:
                    print(f"  ✓ {position}: All position-specific features present")
    
    def prepare_raw_features_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and target (y) for RAW point prediction
        Excludes all cost/value features
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Define columns to exclude
        exclude_cols = [
            self.target_column,
            'player_id',
            'season',
            'feature_generation_time',
            'target_gameweek',
            'actual_points',
            'web_name',
            'team_id'
        ]
        
        # Identify feature columns (all columns except excluded ones)
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        print(f"  Preparing raw features for prediction")
        print(f"  Total features available: {len(feature_cols)}")
        
        # Handle missing values
        X = data[feature_cols].copy()
        y = data[self.target_column].copy()
        
        # Fill NaN values with appropriate defaults
        for col in X.columns:
            if col.startswith('rolling_') or col.endswith('_score'):
                X[col] = X[col].fillna(0)
            elif col.endswith('_efficiency'):
                X[col] = X[col].fillna(1.0)
            elif col.endswith('_overperformance'):
                X[col] = X[col].fillna(0)
            elif col.endswith('_probability') or col.endswith('_chance'):
                X[col] = X[col].fillna(0.5)
            elif col in ['playing_probability', 'status_numeric']:
                X[col] = X[col].fillna(0.5)
            elif col == 'position':
                # Convert position to categorical codes for ML
                position_mapping = {
                    'Goalkeeper': 0,
                    'Defender': 1,
                    'Midfielder': 2,
                    'Forward': 3
                }
                X[col] = X[col].map(position_mapping).fillna(2)  # Default to Midfielder
            else:
                X[col] = X[col].fillna(0)
        
        # Ensure numeric types
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        print(f"  Prepared {len(feature_cols)} raw features and {len(y)} targets")
        
        return X, y
    
    def split_by_position(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data by player position for position-specific models
        
        Args:
            data: DataFrame with 'position' column
            
        Returns:
            Dictionary mapping positions to DataFrames
        """
        if 'position' not in data.columns:
            raise ValueError("Data must contain 'position' column")
        
        # Standardize position names first
        position_mapping = {
            'GKP': 'Goalkeeper',
            'Goalkeeper': 'Goalkeeper',
            'DEF': 'Defender',
            'Defender': 'Defender',
            'MID': 'Midfielder',
            'Midfielder': 'Midfielder',
            'FWD': 'Forward',
            'Forward': 'Forward'
        }
        data = data.copy()
        data['position'] = data['position'].map(position_mapping).fillna(data['position'])
        
        position_data = {}
        
        for position in POSITIONS:
            pos_data = data[data['position'] == position].copy()
            position_data[position] = pos_data
            
            if len(pos_data) > 0:
                print(f"  {position}: {len(pos_data)} samples")
        
        return position_data