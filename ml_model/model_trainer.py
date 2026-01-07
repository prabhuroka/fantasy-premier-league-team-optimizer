"""
Train position-specific ML models for RAW FPL point prediction - FIXED VERSION
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_model.config import (
    MODELS_DIR, LIGHTGBM_PARAMS, XGBOOST_PARAMS, 
    EARLY_STOPPING_ROUNDS, RANDOM_SEED, POSITIONS, DATABASE_PATH,
    MODEL_PREFIX_RAW
)


class ModelTrainer:
    def __init__(self, model_type: str = 'lightgbm', model_prefix: str = MODEL_PREFIX_RAW):
        """
        Initialize ModelTrainer for RAW point prediction
        
        Args:
            model_type: Type of model to train ('lightgbm', 'xgboost', 'random_forest')
            model_prefix: Prefix for model files ('raw_' for raw point models)
        """
        self.model_type = model_type
        self.model_prefix = model_prefix
        self.models_dir = MODELS_DIR
        self.database_path = DATABASE_PATH
        self.models = {}
        self.feature_importances = {}
        self.position_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame = None, y_val: pd.Series = None,
                          position: str = None) -> Any:
        """
        Train a single model for a position - FIXED for LightGBM version
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            position: Position name (for logging)
            
        Returns:
            Trained model
        """
        if position:
            print(f"  Training {self.model_type} model for {position}...")
        
        if len(X_train) < 10:
            print(f"    ⚠ Skipping {position} - insufficient data ({len(X_train)} samples)")
            return None
        
        if self.model_type == 'lightgbm':
            return self._train_lightgbm_fixed(X_train, y_train, X_val, y_val, position)
        elif self.model_type == 'xgboost':
            return self._train_xgboost(X_train, y_train, X_val, y_val, position)
        elif self.model_type == 'random_forest':
            return self._train_random_forest(X_train, y_train, position)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _train_lightgbm_fixed(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            position: str) -> lgb.Booster:
        """Train LightGBM model - FIXED version for compatibility"""
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, valid_data]
        else:
            valid_sets = [train_data]
        
        # Adjust parameters based on position
        params = LIGHTGBM_PARAMS.copy()
        
        # Position-specific adjustments for RAW point prediction
        if position == 'Goalkeeper':
            params['num_leaves'] = 15  # Smaller for goalkeepers
            params['min_child_samples'] = 5  # Smaller for fewer samples
            params['learning_rate'] = 0.03  # Slower learning for more stable predictions
        elif position == 'Defender':
            params['num_leaves'] = 20
            params['min_child_samples'] = 10
            params['learning_rate'] = 0.04
        elif position == 'Midfielder':
            params['num_leaves'] = 31
            params['min_child_samples'] = 20
            params['learning_rate'] = 0.05
        elif position == 'Forward':
            params['num_leaves'] = 25
            params['min_child_samples'] = 10
            params['learning_rate'] = 0.04
        
        # Adjust for small dataset sizes
        n_samples = len(X_train)
        if n_samples < 100:
            params['num_leaves'] = min(params['num_leaves'], 10)
            params['min_child_samples'] = max(2, params['min_child_samples'] // 2)
            params['learning_rate'] = max(0.01, params['learning_rate'] * 0.8)
        
        # Remove problematic parameters for older LightGBM versions
        if 'verbose' in params:
            params.pop('verbose', None)
        
        # Train model with compatibility fix
        try:
            # Try with callbacks first (newer version)
            try:
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=300,
                    valid_sets=valid_sets,
                    callbacks=[
                        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                        lgb.log_evaluation(period=50)
                    ]
                )
            except TypeError:
                # Fall back to older API
                print(f"    Using older LightGBM API for {position}")
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=300,
                    valid_sets=valid_sets,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=50
                )
            
            print(f"    ✓ Trained with {model.current_iteration()} iterations")
            return model
        except Exception as e:
            print(f"    ✗ LightGBM training failed for {position}: {e}")
            
            # Try even simpler approach
            try:
                print(f"    Trying simpler training for {position}")
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=valid_sets
                )
                print(f"    ✓ Trained with {model.current_iteration()} iterations (simple)")
                return model
            except Exception as e2:
                print(f"    ✗ Even simple training failed: {e2}")
                return None
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      position: str) -> xgb.XGBRegressor:
        """Train XGBoost model for RAW point prediction"""
        params = XGBOOST_PARAMS.copy()
        
        # Position-specific adjustments
        if position == 'Goalkeeper':
            params['max_depth'] = 4
            params['learning_rate'] = 0.03
        elif position == 'Defender':
            params['max_depth'] = 5
            params['learning_rate'] = 0.04
        elif position == 'Midfielder':
            params['max_depth'] = 6
            params['learning_rate'] = 0.05
        elif position == 'Forward':
            params['max_depth'] = 5
            params['learning_rate'] = 0.04
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        else:
            watchlist = [(dtrain, 'train')]
        
        # Train model
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=300,
                evals=watchlist,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=50
            )
            return model
        except Exception as e:
            print(f"    ✗ XGBoost training failed for {position}: {e}")
            return None
    
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            position: str) -> RandomForestRegressor:
        """Train Random Forest model for RAW point prediction"""
        # Adjust parameters based on position
        n_estimators = 100  # Reduced for smaller datasets
        max_depth = 10
        
        if position == 'Goalkeeper':
            n_estimators = 50
            max_depth = 8
        elif position == 'Defender':
            n_estimators = 75
            max_depth = 9
        elif position == 'Midfielder':
            n_estimators = 100
            max_depth = 10
        elif position == 'Forward':
            n_estimators = 75
            max_depth = 9
        
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=0
            )
            
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"    ✗ Random Forest training failed for {position}: {e}")
            return None
    
    def train_position_models_raw(self, position_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                                selected_features: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Train separate models for each position for RAW point prediction
        
        Args:
            position_data: Dictionary mapping positions to (X, y) tuples
            selected_features: Dictionary mapping positions to selected feature lists
            
        Returns:
            Dictionary of trained models
        """
        position_models = {}
        
        for position, (X, y) in position_data.items():
            if position in selected_features and len(X) > 10:
                # Use selected features for this position
                features = selected_features[position]
                
                # Ensure features exist in X
                available_features = [f for f in features if f in X.columns]
                if not available_features:
                    print(f"  ⚠ No available features for {position}")
                    continue
                
                X_selected = X[available_features].copy()
                
                # Split into train/validation (80/20)
                split_idx = int(len(X_selected) * 0.8)
                X_train = X_selected.iloc[:split_idx]
                y_train = y.iloc[:split_idx]
                X_val = X_selected.iloc[split_idx:] if split_idx < len(X_selected) else None
                y_val = y.iloc[split_idx:] if split_idx < len(y) else None
                
                print(f"  {position}: Training on {len(X_train)} samples, {len(available_features)} features")
                if X_val is not None:
                    print(f"          Validating on {len(X_val)} samples")
                
                # Train model
                model = self.train_single_model(X_train, y_train, X_val, y_val, position)
                
                if model is not None:
                    position_models[position] = {
                        'model': model,
                        'features': available_features,
                        'n_samples': len(X),
                        'n_train': len(X_train),
                        'n_val': len(X_val) if X_val is not None else 0,
                        'model_type': self.model_type
                    }
                    
                    # Store feature importance if available
                    if hasattr(model, 'feature_importance'):
                        try:
                            importance = pd.DataFrame({
                                'feature': available_features,
                                'importance': model.feature_importance(importance_type='gain')
                            })
                            self.feature_importances[position] = importance.sort_values('importance', ascending=False)
                            print(f"    ✓ Feature importance calculated")
                        except:
                            print(f"    ⚠ Could not calculate feature importance")
                            pass
                    
                    # Calculate validation metrics if validation data exists
                    if X_val is not None and y_val is not None and len(X_val) > 0:
                        try:
                            if self.model_type == 'lightgbm':
                                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                            elif self.model_type == 'xgboost':
                                import xgboost as xgb
                                dval = xgb.DMatrix(X_val)
                                y_pred = model.predict(dval)
                            else:
                                y_pred = model.predict(X_val)
                            
                            mae = np.mean(np.abs(y_val - y_pred))
                            print(f"    ✓ Validation MAE: {mae:.3f}")
                        except:
                            print(f"    ⚠ Could not calculate validation metrics")
        
        self.position_models = position_models
        print(f"\n✓ Successfully trained {len(position_models)} position models for RAW point prediction")
        
        return position_models
    
    def save_models(self, prefix: str = None):
        """Save trained models to disk with appropriate prefix - FIXED version"""
        if prefix is None:
            prefix = self.model_prefix
        
        if not self.position_models:
            print("  ⚠ No models to save")
            return
        
        print(f"\nSaving {len(self.position_models)} raw point models...")
        
        for position, model_info in self.position_models.items():
            model = model_info['model']
            features = model_info['features']
            
            if model is None:
                continue
            
            # Convert position to abbreviation for filenames
            position_abbrev = {
                'Goalkeeper': 'GKP',
                'Defender': 'DEF',
                'Midfielder': 'MID',
                'Forward': 'FWD'
            }.get(position, position)
            
            # Save model
            if self.model_type == 'lightgbm':
                model_file = os.path.join(self.models_dir, f'{prefix}{position_abbrev}_lgb_model.txt')
                try:
                    model.save_model(model_file)
                    print(f"  ✓ Saved {position} model to {model_file}")
                except Exception as e:
                    print(f"  ✗ Failed to save {position} model: {e}")
                    continue
            elif self.model_type == 'xgboost':
                model_file = os.path.join(self.models_dir, f'{prefix}{position_abbrev}_xgb_model.json')
                try:
                    model.save_model(model_file)
                    print(f"  ✓ Saved {position} model to {model_file}")
                except Exception as e:
                    print(f"  ✗ Failed to save {position} model: {e}")
                    continue
            else:
                model_file = os.path.join(self.models_dir, f'{prefix}{position_abbrev}_rf_model.pkl')
                try:
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"  ✓ Saved {position} model to {model_file}")
                except Exception as e:
                    print(f"  ✗ Failed to save {position} model: {e}")
                    continue
            
            # Save features
            features_file = os.path.join(self.models_dir, f'{prefix}{position_abbrev}_features.pkl')
            try:
                with open(features_file, 'wb') as f:
                    pickle.dump(features, f)
                print(f"  ✓ Saved {position} features ({len(features)} features)")
            except Exception as e:
                print(f"  ✗ Failed to save {position} features: {e}")
            
            # Save feature importance if available
            if position in self.feature_importances:
                importance_file = os.path.join(self.models_dir, f'{prefix}{position_abbrev}_feature_importance.pkl')
                try:
                    with open(importance_file, 'wb') as f:
                        pickle.dump(self.feature_importances[position], f)
                    print(f"  ✓ Saved {position} feature importance")
                except Exception as e:
                    print(f"  ⚠ Could not save {position} feature importance: {e}")
        
        print(f"\n✓ All raw point models saved to {self.models_dir}")
        
        # Save configuration
        try:
            config = {
                'model_type': self.model_type,
                'model_prefix': prefix,
                'positions_trained': list(self.position_models.keys()),
                'feature_counts': {pos: len(info['features']) for pos, info in self.position_models.items()},
                'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'lightgbm_params': LIGHTGBM_PARAMS
            }
            
            config_file = os.path.join(self.models_dir, f'{prefix}model_config.pkl')
            with open(config_file, 'wb') as f:
                pickle.dump(config, f)
            
            print(f"✓ Model configuration saved to {config_file}")
        except Exception as e:
            print(f"⚠ Could not save model configuration: {e}")
    
    def load_models(self, prefix: str = None):
        """Load trained models from disk - FIXED version"""
        if prefix is None:
            prefix = self.model_prefix
        
        print(f"Loading raw point models with prefix '{prefix}'...")
        
        position_mapping = {
            'GKP': 'Goalkeeper',
            'DEF': 'Defender',
            'MID': 'Midfielder',
            'FWD': 'Forward'
        }
        
        loaded_count = 0
        
        for abbrev, position in position_mapping.items():
            model_info = {}
            
            # Check if model file exists
            if self.model_type == 'lightgbm':
                model_file = os.path.join(self.models_dir, f'{prefix}{abbrev}_lgb_model.txt')
            elif self.model_type == 'xgboost':
                model_file = os.path.join(self.models_dir, f'{prefix}{abbrev}_xgb_model.json')
            else:
                model_file = os.path.join(self.models_dir, f'{prefix}{abbrev}_rf_model.pkl')
            
            if not os.path.exists(model_file):
                print(f"  ⚠ Model file not found: {model_file}")
                continue
            
            # Try to load model
            try:
                if self.model_type == 'lightgbm':
                    model_info['model'] = lgb.Booster(model_file=model_file)
                    print(f"  ✓ Loaded {position} model from {model_file}")
                elif self.model_type == 'xgboost':
                    model_info['model'] = xgb.Booster()
                    model_info['model'].load_model(model_file)
                    print(f"  ✓ Loaded {position} model from {model_file}")
                else:
                    with open(model_file, 'rb') as f:
                        model_info['model'] = pickle.load(f)
                    print(f"  ✓ Loaded {position} model from {model_file}")
            except Exception as e:
                print(f"  ✗ Failed to load {position} model: {e}")
                continue
            
            # Load features
            features_file = os.path.join(self.models_dir, f'{prefix}{abbrev}_features.pkl')
            if os.path.exists(features_file):
                try:
                    with open(features_file, 'rb') as f:
                        model_info['features'] = pickle.load(f)
                    print(f"  ✓ Loaded {len(model_info['features'])} features for {position}")
                except Exception as e:
                    print(f"  ✗ Failed to load {position} features: {e}")
                    continue
            else:
                print(f"  ⚠ Features file not found: {features_file}")
                continue
            
            if 'model' in model_info and 'features' in model_info:
                self.position_models[position] = model_info
                loaded_count += 1
        
        if loaded_count > 0:
            print(f"\n✓ Successfully loaded {loaded_count} position models")
        else:
            print(f"\n✗ No models loaded")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of trained models
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'total_models': len(self.position_models),
            'model_type': self.model_type,
            'model_prefix': self.model_prefix,
            'positions': {},
            'total_samples': 0,
            'total_features': 0
        }
        
        for position, model_info in self.position_models.items():
            summary['positions'][position] = {
                'n_samples': model_info.get('n_samples', 0),
                'n_features': len(model_info.get('features', [])),
                'n_train': model_info.get('n_train', 0),
                'n_val': model_info.get('n_val', 0)
            }
            
            summary['total_samples'] += model_info.get('n_samples', 0)
            summary['total_features'] += len(model_info.get('features', []))
        
        return summary