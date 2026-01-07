"""
Generate RAW point predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import warnings
import sqlite3
warnings.filterwarnings('ignore')

from data_pipeline.database import FPLDatabase
from feature_engineering.feature_builder import FeatureBuilder
from ml_model.config import PREDICTIONS_DIR, MODELS_DIR, DATABASE_PATH, EXCLUDED_FEATURES_RAW
import lightgbm as lgb
import pickle
from datetime import datetime


class RawPredictionGenerator:
    def __init__(self, db_path: str = DATABASE_PATH, model_type: str = 'lightgbm'):
        """
        Initialize RawPredictionGenerator for RAW point predictions - FIXED
        
        Args:
            db_path: Path to database
            model_type: Type of model to use ('lightgbm', 'xgboost')
        """
        self.db_path = db_path
        self.model_type = model_type
        self.models_dir = MODELS_DIR
        self.predictions_dir = PREDICTIONS_DIR
        
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Load raw point models
        self.position_models = self._load_raw_models_fixed()
    
    def get_latest_gameweek(self) -> int:
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
                WHERE season = '2025-2026'
            """
            result = pd.read_sql_query(query, conn)
            conn.close()
            
            if result is not None and not result.empty and result.iloc[0]['latest_gw']:
                return int(result.iloc[0]['latest_gw'])
            return 1  # Default to GW1 if no data
            
        except Exception as e:
            print(f"Error getting latest gameweek: {e}")
            return 1
    
    def get_next_gameweek(self) -> int:
        """
        Get the next gameweek to predict (latest + 1)
        
        Returns:
            Next gameweek number
        """
        latest_gw = self.get_latest_gameweek()
        return latest_gw + 1
    
    def _load_raw_models_fixed(self) -> Dict[str, Dict]:
        """Load trained RAW point models from disk - FIXED version"""
        position_models = {}
        
        position_mapping = {
            'GKP': 'Goalkeeper',
            'DEF': 'Defender',
            'MID': 'Midfielder',
            'FWD': 'Forward'
        }
        
        model_prefix = 'raw_'  # Raw point model prefix
        
        print("Loading raw point models...")
        
        for abbrev, position in position_mapping.items():
            model_info = {}
            
            # Try to load raw point model
            model_file = os.path.join(self.models_dir, f'{model_prefix}{abbrev}_lgb_model.txt')
            
            if not os.path.exists(model_file):
                print(f"  ⚠ Model file not found: {model_file}")
                # Try without prefix
                model_file_alt = os.path.join(self.models_dir, f'{abbrev}_lgb_model.txt')
                if os.path.exists(model_file_alt):
                    model_file = model_file_alt
                    print(f"  Found alternative model file: {model_file_alt}")
                else:
                    print(f"  ⚠ No model file found for {position}")
                    continue
            
            try:
                model_info['model'] = lgb.Booster(model_file=model_file)
                print(f"  ✓ Loaded {position} model from {model_file}")
            except Exception as e:
                print(f"  ✗ Failed to load {position} model: {e}")
                continue
            
            # Load features
            features_file = os.path.join(self.models_dir, f'{model_prefix}{abbrev}_features.pkl')
            
            if not os.path.exists(features_file):
                print(f"  ⚠ Features file not found: {features_file}")
                # Try without prefix
                features_file_alt = os.path.join(self.models_dir, f'{abbrev}_features.pkl')
                if os.path.exists(features_file_alt):
                    features_file = features_file_alt
                    print(f"  Found alternative features file: {features_file_alt}")
                else:
                    print(f"  ⚠ No features file found for {position}")
                    continue
            
            try:
                with open(features_file, 'rb') as f:
                    model_info['features'] = pickle.load(f)
                print(f"  ✓ Loaded {len(model_info['features'])} features for {position}")
            except Exception as e:
                print(f"  ✗ Failed to load {position} features: {e}")
                continue
            
            if 'model' in model_info and 'features' in model_info:
                position_models[position] = model_info
                print(f"  ✓ Successfully loaded {position} model with {len(model_info['features'])} features")
        
        print(f"\nLoaded raw point models for {len(position_models)} positions")
        
        # Print loaded models
        for position, info in position_models.items():
            print(f"  {position}: {len(info['features'])} features")
            # Show first few features
            if info['features']:
                print(f"    Features sample: {info['features'][:5]}")
        
        return position_models
    
    def prepare_features_for_prediction_fixed(self, gameweek: int, season: str = "2025-2026") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for prediction - FIXED version with 5+ gameweeks for rolling features
        
        Args:
            gameweek: Gameweek to prepare features for
            season: Season string
            
        Returns:
            Tuple of (features_df, cost_data)
        """
        print(f"  Preparing features for GW{gameweek}...")
        
        # Connect to database
        db = FPLDatabase(self.db_path)
        if not db.connect():
            raise ConnectionError("Failed to connect to database")
        
        try:
            # Use previous gameweek as target
            previous_gw = max(1, gameweek - 1)
            
            # Determine max rolling window needed by checking model features
            max_rolling_window_needed = 5  # Default
        
            # Check if models need 10-game rolling features
            for position, model_info in self.position_models.items():
                features = model_info['features']
                for feature in features:
                    if '10' in feature and 'rolling' in feature:
                        max_rolling_window_needed = 10
                        break
                    if '5' in feature and 'rolling' in feature:
                        max_rolling_window_needed = max(max_rolling_window_needed, 5)
            
            print(f"    Models need up to {max_rolling_window_needed}-game rolling features")
        
            # Get enough history for the max rolling window
            min_history_gw = max(1, previous_gw - max_rolling_window_needed)
            
            print(f"    Using GW{min_history_gw}-{previous_gw} data ({max_rolling_window_needed}+ gameweeks)")
            
            history_query = """
                SELECT 
                    pgs.id as player_id,
                    pgs.gw as gameweek,
                    pgs.season,
                    pgs.minutes,
                    pgs.total_points,
                    pgs.goals_scored,
                    pgs.assists,
                    pgs.clean_sheets,
                    pgs.goals_conceded,
                    pgs.expected_goals,
                    pgs.expected_assists,
                    pgs.expected_goal_involvements,
                    pgs.expected_goals_conceded,
                    pgs.saves,
                    pgs.bonus,
                    pgs.selected_by_percent,
                    pgs.now_cost,
                    pgs.tackles,
                    pgs.clearances_blocks_interceptions,
                    pgs.influence,
                    pgs.creativity,
                    pgs.threat,
                    pgs.ict_index,
                    pgs.bps,
                    pgs.yellow_cards,
                    pgs.red_cards,
                    pgs.own_goals,
                    pgs.penalties_saved,
                    pgs.penalties_missed,
                    pgs.form
                FROM player_gameweek_stats pgs
                WHERE pgs.gw >= ? 
                  AND pgs.gw <= ?
                  AND pgs.season = ?
                ORDER BY pgs.id, pgs.gw
            """
            
            # Get data for the last 5+ gameweeks
            history_data = db.execute_query(history_query, (min_history_gw, previous_gw, season))
            
            if history_data is None or history_data.empty:
                print(f"    ⚠ No history data found for GW{min_history_gw}-{previous_gw}")
                # Fallback: get all available data
                fallback_query = """
                    SELECT * FROM player_gameweek_stats 
                    WHERE season = ? AND gw <= ?
                    ORDER BY id, gw DESC
                """
                history_data = db.execute_query(fallback_query, (season, previous_gw))
                
            if history_data is None or history_data.empty:
                print("    ⚠ No history data found even with fallback")
                return pd.DataFrame(), pd.DataFrame()
            
            print(f"    Loaded {len(history_data)} history records from {history_data['gameweek'].min()} to {history_data['gameweek'].max()}")
            
            # Get unique players count
            unique_players = history_data['player_id'].nunique()
            print(f"    Unique players in history: {unique_players}")
            
            # Get player metadata
            metadata_query = """
                SELECT 
                    player_id,
                    COALESCE(position, 'UNK') as position,
                    COALESCE(team_code, 0) as team_id,
                    COALESCE(web_name, 'Unknown') as web_name
                FROM players
                WHERE season = ?
            """
            
            player_data = db.execute_query(metadata_query, (season,))
            
            if player_data is None or player_data.empty:
                print("    ⚠ No player metadata found")
                return pd.DataFrame(), pd.DataFrame()
            
            print(f"    Loaded metadata for {len(player_data)} players")
            
            # Get player costs from LATEST gameweek
            cost_query = """
                SELECT 
                    id as player_id,
                    now_cost
                FROM player_gameweek_stats 
                WHERE gw = ? AND season = ?
            """
            cost_data = db.execute_query(cost_query, (previous_gw, season))
            
            if cost_data is not None and not cost_data.empty:
                print(f"    Loaded costs for {len(cost_data)} players")
            
            # Create features using feature builder
            feature_builder = FeatureBuilder(db)
            
            features_df = feature_builder.create_comprehensive_features(
                player_history=history_data,
                player_info=player_data,
                gameweek=previous_gw,  # Use the latest GW for feature calculation
                season=season
            )
            
            if features_df.empty:
                print("    ⚠ No features generated")
                return pd.DataFrame(), pd.DataFrame()
            
            print(f"    Generated features for {len(features_df)} players")
            
            # Debug: Check if rolling features are present
            rolling_features = [col for col in features_df.columns if 'rolling_' in col]
            print(f"    Rolling features generated: {len(rolling_features)}")
            if rolling_features:
                print(f"    Sample rolling features: {rolling_features[:5]}")
            
            # Remove duplicate player entries
            if 'player_id' in features_df.columns:
                initial_count = len(features_df)
                features_df = features_df.drop_duplicates(subset=['player_id'], keep='last')
                removed_count = initial_count - len(features_df)
                if removed_count > 0:
                    print(f"    Removed {removed_count} duplicate player entries")
            
            # Remove cost/value features for raw point prediction
            existing_exclude = [f for f in EXCLUDED_FEATURES_RAW if f in features_df.columns]
            if existing_exclude:
                print(f"    Removing {len(existing_exclude)} cost/value features")
                features_df = features_df.drop(columns=existing_exclude)
            
            print(f"    Final feature set: {len(features_df)} unique players, {len(features_df.columns)} features")
            
            return features_df, cost_data
            
        finally:
            db.close()       
    
    def predict_next_gameweek_fixed(self, next_gw: int = None) -> pd.DataFrame:
        """
        Generate RAW point predictions for next gameweek - FIXED version
        
        Args:
            next_gw: Next gameweek number (None to auto-detect)
            
        Returns:
            DataFrame with raw point predictions
        """
        # Determine next gameweek if not provided
        if next_gw is None:
            next_gw = self.get_next_gameweek()
        
        print(f"\n{'='*60}")
        print(f"GENERATING RAW POINT PREDICTIONS FOR GW{next_gw}")
        print(f"{'='*60}")
        
        # Check if models are loaded
        if not self.position_models:
            print("  ⚠ No raw point models loaded. Trying to load again...")
            self.position_models = self._load_raw_models_fixed()
            
            if not self.position_models:
                print("  ✗ Failed to load raw point models")
                return pd.DataFrame()
        
        print(f"  Using {len(self.position_models)} position models")
        
        # Prepare features
        print("\n1. Preparing features for raw point prediction...")
        
        try:
            features_df, cost_data = self.prepare_features_for_prediction_fixed(next_gw)
            
            if features_df.empty:
                print("  ⚠ No features generated")
                return pd.DataFrame()
            
            self.validate_feature_consistency(features_df)
            
            print(f"  ✓ Generated features for {len(features_df)} players")
            print(f"  ✓ Total features: {len(features_df.columns)}")
            
            # Standardize position names
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
            
            if 'position' in features_df.columns:
                features_df['position'] = features_df['position'].map(position_mapping).fillna(features_df['position'])
                print(f"  ✓ Standardized position names")
            
        except Exception as e:
            print(f"  ✗ Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        
        # Make predictions for each position
        all_predictions = []
        
        for position, model_info in self.position_models.items():
            # Filter players by position
            pos_players = features_df[features_df['position'] == position].copy()
            
            if len(pos_players) == 0:
                print(f"  ⚠ No {position} players found in features")
                continue
            
            # Get model and raw features
            model = model_info['model']
            required_features = model_info['features']
            
            print(f"\n  Predicting for {position}...")
            print(f"    Model expects {len(required_features)} features")
            print(f"    Found {len(pos_players)} {position} players")
            
            # Find which required features are available
            available_features = [f for f in required_features if f in pos_players.columns]
            missing_features = [f for f in required_features if f not in pos_players.columns]
            
            if missing_features:
                missing_count = len(missing_features)
                available_ratio = len(available_features) / len(required_features)
                
                print(f"    ⚠ Missing {missing_count} features ({available_ratio:.1%} available)")
            
                # Critical threshold check
                if available_ratio < 0.7:  # Less than 70% features available
                    print(f"    ✗ CRITICAL: Skipping {position} - insufficient features")
                    continue
                elif missing_count <= 5:
                    print(f"      Missing: {missing_features}")
                else:
                    print(f"      First 5 missing: {missing_features[:5]}")
                    
                # Attempt feature imputation for important features
                critical_features = ['playing_probability', 'form_adjusted', 'status_numeric', 
                                   'rolling_points_3', 'expected_goals']
                missing_critical = [f for f in missing_features if f in critical_features]
                
                if missing_critical:
                    print(f"    ⚠ Missing critical features: {missing_critical}")
                    # Attempt imputation based on position averages
                    self.impute_critical_features(pos_players, missing_critical, position)
                    # Re-check availability after imputation
                    available_features = [f for f in required_features if f in pos_players.columns]
            
            if not available_features:
                print(f"    ⚠ No matching features for {position}")
                continue
            
            print(f"    Using {len(available_features)} available features")
            
            # Prepare features with consistent ordering
            X_pred = pos_players[available_features].copy()
            
            # Fill missing values
            X_pred = X_pred.fillna(0)
            
            # Ensure numeric types
            for col in available_features:
                X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')
            X_pred = X_pred.fillna(0)
            
            # Ensure the feature order matches training
            X_pred = X_pred[available_features]
            
            # Make RAW point predictions
            try:
                print(f"    Making predictions...")
                
                if self.model_type == 'lightgbm':
                    raw_predictions = model.predict(X_pred)
                    feature_importance_per_player = self.get_feature_importance_per_player(
                    model, X_pred, available_features
                    )
                    confidence_scores = self.calculate_prediction_confidence(
                        model, X_pred, available_features, required_features
                    )
                elif self.model_type == 'xgboost':
                    import xgboost as xgb
                    dpred = xgb.DMatrix(X_pred)
                    raw_predictions = model.predict(dpred)
                else:
                    raw_predictions = model.predict(X_pred)
                
                print(f"    ✓ Generated predictions for {len(raw_predictions)} players")
                print(f"    Prediction range: {raw_predictions.min():.1f} - {raw_predictions.max():.1f}")
                
                # Create prediction DataFrame
                pos_predictions = pos_players[['player_id', 'position']].copy()
                if feature_importance_per_player is not None:
                    pos_predictions['top_features'] = feature_importance_per_player
                    print(f"    ✓ Added feature importance analysis")
                
                if confidence_scores is not None:
                    pos_predictions['prediction_confidence'] = confidence_scores
                    
                    # Categorize confidence
                    pos_predictions['confidence_category'] = pd.cut(
                        confidence_scores,
                        bins=[0, 0.6, 0.8, 0.9, 1.0],
                        labels=['Low', 'Medium', 'High', 'Very High']
                    )
                    print(f"    ✓ Added confidence scores (avg: {np.mean(confidence_scores):.2f})")
                
                # Add additional info if available
                if 'web_name' in pos_players.columns:
                    pos_predictions['web_name'] = pos_players['web_name']
                if 'team_id' in pos_players.columns:
                    pos_predictions['team_id'] = pos_players['team_id']
                
                # Add cost if available
                if cost_data is not None and not cost_data.empty:
                    cost_data['player_id'] = cost_data['player_id'].astype(str)
                    pos_predictions['player_id'] = pos_predictions['player_id'].astype(str)
                    
                    pos_predictions = pos_predictions.merge(
                        cost_data[['player_id', 'now_cost']],
                        on='player_id',
                        how='left'
                    )
                    print(f"    Added cost data for {len(pos_predictions)} players")
                
                pos_predictions['raw_predicted_points'] = raw_predictions
                pos_predictions['predicted_points'] = raw_predictions  # For compatibility
                pos_predictions['gameweek'] = next_gw
                pos_predictions['season'] = "2025-2026"
                pos_predictions['prediction_timestamp'] = datetime.now()
                pos_predictions['model_type'] = self.model_type
                pos_predictions['prediction_type'] = 'raw_points'
                pos_predictions['features_used'] = len(available_features)
                
                # Add confidence intervals
                pos_predictions['predicted_min'] = np.maximum(0, raw_predictions - 2)
                pos_predictions['predicted_max'] = raw_predictions + 2
                
                all_predictions.append(pos_predictions)
                
            except Exception as e:
                print(f"    ✗ Prediction failed for {position}: {e}")
                continue
        
        # Combine all predictions
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            
                    # FIX: Remove duplicate predictions for the same player
            if 'player_id' in predictions_df.columns:
                initial_count = len(predictions_df)
                predictions_df = predictions_df.drop_duplicates(subset=['player_id'], keep='first')
                removed_count = initial_count - len(predictions_df)
                if removed_count > 0:
                    print(f"  Removed {removed_count} duplicate player predictions")
            
            # Sort by raw predicted points
            predictions_df = predictions_df.sort_values('raw_predicted_points', ascending=False)
            
            # Add rank
            predictions_df['rank_raw'] = range(1, len(predictions_df) + 1)
            
            # Add value metrics (for optimization phase)
            if 'now_cost' in predictions_df.columns:
                predictions_df['now_cost_millions'] = predictions_df['now_cost'] / 10
                predictions_df['points_per_million'] = predictions_df.apply(
                    lambda row: row['raw_predicted_points'] / row['now_cost_millions'] 
                    if row['now_cost_millions'] > 0 else 0,
                    axis=1
                )
                
                # Add value rank
                value_sorted = predictions_df.sort_values('points_per_million', ascending=False)
                value_sorted['rank_value'] = range(1, len(value_sorted) + 1)
                
                # Merge back
                predictions_df = predictions_df.merge(
                    value_sorted[['player_id', 'rank_value']],
                    on='player_id',
                    how='left'
                )
                
                # Resort by raw points
                predictions_df = predictions_df.sort_values('raw_predicted_points', ascending=False)
            
            # Save predictions
            output_file = os.path.join(self.predictions_dir, f'raw_points_predictions_gw{next_gw}.csv')
            predictions_df.to_csv(output_file, index=False)
            print(f"\n✓ Generated {len(predictions_df)} RAW point predictions for GW{next_gw}")
            print(f"  ✓ Saved predictions to: {output_file}")
            
            
            # Show prediction summary
            self._show_prediction_summary(predictions_df)
            
            return predictions_df
        
        print("  ✗ No predictions generated")
        return pd.DataFrame()
    
    def validate_feature_consistency(self, generated_features: pd.DataFrame):
        """
        Validate that generated features match what models expect
        """
        print("\n  Validating feature consistency...")
        
        missing_summary = {}
        
        for position, model_info in self.position_models.items():
            if 'features' not in model_info:
                continue
                
            required_features = model_info['features']
            available_features = [f for f in required_features if f in generated_features.columns]
            missing_features = set(required_features) - set(available_features)
            
            if missing_features:
                missing_summary[position] = {
                    'missing_count': len(missing_features),
                    'missing_features': list(missing_features)[:5],  # Show first 5
                    'available_ratio': len(available_features) / len(required_features)
                }
        
        if missing_summary:
            print(f"  ⚠ Feature consistency issues found:")
            for position, info in missing_summary.items():
                print(f"    {position}: {info['missing_count']} missing features "
                      f"({info['available_ratio']:.1%} available)")
                if info['missing_count'] > 5:
                    print(f"      First 5 missing: {info['missing_features']}")
        else:
            print(f"  ✓ All features available for all positions")
        
        return missing_summary
    
    def calculate_prediction_confidence(self, model, X_pred: pd.DataFrame, 
                                  available_features: List[str], 
                                  required_features: List[str]) -> np.ndarray:
        """
        Calculate confidence score for each prediction based on:
        1. Feature availability ratio
        2. Model certainty (prediction variance if available)
        3. Data quality (missing values, outliers)
        """
        try:
            n_players = len(X_pred)
            confidence_scores = np.zeros(n_players)
            
            # 1. Feature availability component (40% weight)
            feature_ratio = len(available_features) / len(required_features)
            feature_score = np.clip(feature_ratio, 0.3, 1.0)  # Minimum 0.3 even with few features
            
            # 2. Data quality component (30% weight)
            missing_ratio = X_pred.isnull().sum(axis=1) / len(X_pred.columns)
            outlier_ratio = self.detect_outliers(X_pred)
            data_quality_score = 1.0 - (missing_ratio * 0.5 + outlier_ratio * 0.5)
            
            # 3. Model certainty component (30% weight) - if LightGBM
            if self.model_type == 'lightgbm' and hasattr(model, 'predict'):
                # Use prediction std if available (from multiple trees)
                try:
                    # Try to get leaf indices to estimate uncertainty
                    leaf_preds = model.predict(X_pred, pred_leaf=True)
                    # Calculate variance across trees (simplified)
                    tree_variance = np.std(leaf_preds, axis=1)
                    model_certainty = 1.0 / (1.0 + tree_variance)
                    model_certainty = np.clip(model_certainty, 0.5, 1.0)
                except:
                    model_certainty = np.ones(n_players) * 0.8  # Default
            else:
                model_certainty = np.ones(n_players) * 0.8
            
            # Combine scores with weights
            confidence_scores = (
                feature_score * 0.4 + 
                data_quality_score * 0.3 + 
                model_certainty * 0.3
            )
            
            # Ensure scores are between 0.3 and 1.0
            confidence_scores = np.clip(confidence_scores, 0.3, 1.0)
            
            return confidence_scores
            
        except Exception as e:
            print(f"    ⚠ Confidence calculation failed: {e}")
            # Fallback: simple feature ratio
            feature_ratio = len(available_features) / len(required_features)
            return np.ones(n_players) * np.clip(feature_ratio, 0.5, 1.0)

    def detect_outliers(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detect outliers in features using IQR method
        Returns ratio of outlier features per player
        """
        try:
            outlier_ratios = np.zeros(len(X))
            
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:  # Avoid division by zero
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Mark outliers
                        outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
                        outlier_ratios[outliers] += 1
            
            # Convert to ratio
            outlier_ratios = outlier_ratios / len(X.columns)
            return outlier_ratios
            
        except:
            return np.zeros(len(X))
    
    def impute_critical_features(self, data: pd.DataFrame, missing_features: List[str], 
                           position: str) -> pd.DataFrame:
        """
        Impute critical missing features with position-based defaults
        """
        print(f"    Imputing {len(missing_features)} critical features...")
        
        position_defaults = {
            'Goalkeeper': {
                'playing_probability': 0.8,
                'form_adjusted': 3.0,
                'status_numeric': 1.0,
                'rolling_points_3': 3.0,
                'expected_goals': 0.0,
                'expected_assists': 0.0
            },
            'Defender': {
                'playing_probability': 0.85,
                'form_adjusted': 3.5,
                'status_numeric': 1.0,
                'rolling_points_3': 3.5,
                'expected_goals': 0.1,
                'expected_assists': 0.1
            },
            'Midfielder': {
                'playing_probability': 0.8,
                'form_adjusted': 4.0,
                'status_numeric': 1.0,
                'rolling_points_3': 4.0,
                'expected_goals': 0.2,
                'expected_assists': 0.3
            },
            'Forward': {
                'playing_probability': 0.75,
                'form_adjusted': 4.5,
                'status_numeric': 1.0,
                'rolling_points_3': 4.5,
                'expected_goals': 0.4,
                'expected_assists': 0.1
            }
        }
        
        defaults = position_defaults.get(position, position_defaults['Midfielder'])
        
        for feature in missing_features:
            if feature in defaults:
                data[feature] = defaults[feature]
                print(f"      Imputed {feature} = {defaults[feature]}")
            elif feature.startswith('rolling_'):
                data[feature] = 0.0
                print(f"      Imputed {feature} = 0.0")
            elif 'probability' in feature or 'chance' in feature:
                data[feature] = 0.5
                print(f"      Imputed {feature} = 0.5")
            else:
                data[feature] = 0.0
                print(f"      Imputed {feature} = 0.0 (generic)")
        
        return data
    
    def get_feature_importance_per_player(self, model, X_pred: pd.DataFrame, 
                                    feature_names: List[str]) -> List[str]:
        """
        Get top contributing features for each prediction
        """
        try:
            if self.model_type != 'lightgbm':
                return None
                
            # Get SHAP values for feature importance
            import shap
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_pred)
            
            # Get top 3 features for each player
            top_features_list = []
            for i in range(len(X_pred)):
                # Get absolute SHAP values for this prediction
                player_shap = np.abs(shap_values[i])
                
                # Get indices of top 3 features
                top_indices = np.argsort(player_shap)[-3:][::-1]
                top_features = [feature_names[idx] for idx in top_indices]
                
                # Format as string
                top_features_str = " | ".join([f"{feat}" for feat in top_features])
                top_features_list.append(top_features_str)
            
            return top_features_list
        
        except Exception as e:
            print(f"    ⚠ Could not calculate per-player feature importance: {e}")
            # Fallback: global feature importance
            try:
                if hasattr(model, 'feature_importance'):
                    importance = model.feature_importance(importance_type='gain')
                    top_indices = np.argsort(importance)[-3:][::-1]
                    top_features = [feature_names[idx] for idx in top_indices]
                    return [" | ".join(top_features)] * len(X_pred)
            except:
                pass
            
            return None
        
    def _show_prediction_summary(self, predictions_df: pd.DataFrame, top_n: int = 10):
        """Display prediction summary"""
        if predictions_df.empty:
            return
        
        print(f"\nPREDICTION SUMMARY:")
        print("="*80)
        
        # Top by raw points
        print(f"\nTOP {top_n} PLAYERS BY RAW POINT PREDICTION:")
        print(f"{'Rank':<5} {'Player':<20} {'Position':<12} {'Raw Points':>12} {'Cost (£M)':>10} {'Value':>10}")
        print("-" * 80)
        
        top_raw = predictions_df.head(top_n)
        for idx, row in top_raw.iterrows():
            player_name = row.get('web_name', f"Player {row['player_id']}")
            position = row.get('position', 'Unknown')
            raw_points = row['raw_predicted_points']
            cost = row.get('now_cost_millions', 0) if 'now_cost_millions' in row else row.get('now_cost', 0) / 10
            value = row.get('points_per_million', 0)
            
            print(f"{row['rank_raw']:<5} {player_name:<20} {position:<12} {raw_points:>11.1f} {cost:>9.1f} {value:>9.2f}")
        
        # Position breakdown
        print(f"\nPOSITION BREAKDOWN:")
        for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            pos_data = predictions_df[predictions_df['position'] == position]
            if not pos_data.empty:
                avg_points = pos_data['raw_predicted_points'].mean()
                avg_cost = pos_data.get('now_cost_millions', pd.Series([0])).mean()
                avg_value = pos_data.get('points_per_million', pd.Series([0])).mean()
                
                print(f"  {position}: {len(pos_data)} players")
                print(f"    Avg points: {avg_points:.2f}, Avg cost: £{avg_cost:.1f}M, Avg value: {avg_value:.2f} pts/£M")
    
    def predict_next_gameweek(self, next_gw: int = None) -> pd.DataFrame:
        """Wrapper for fixed prediction method"""
        return self.predict_next_gameweek_fixed(next_gw)