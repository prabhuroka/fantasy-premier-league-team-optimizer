"""
Feature selection methods for RAW point prediction models
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

from ml_model.config import MAX_FEATURES_PER_MODEL, MIN_FEATURE_IMPORTANCE


class FeatureSelector:
    def __init__(self, max_features: int = MAX_FEATURES_PER_MODEL, 
                 min_importance: float = MIN_FEATURE_IMPORTANCE):
        """
        Initialize FeatureSelector for RAW point prediction
        
        Args:
            max_features: Maximum number of features to select
            min_importance: Minimum importance threshold
        """
        self.max_features = max_features
        self.min_importance = min_importance
        
    def select_features_position_aware(self, X: pd.DataFrame, y: pd.Series,
                                      position: str) -> Tuple[List[str], Dict]:
        """
        POSITION-AWARE feature selection for RAW point prediction
        
        Args:
            X: Feature DataFrame
            y: Target Series
            position: Player position
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        print(f"  Selecting features for {position}...")
        
        # Position-specific feature priorities
        position_priorities = {
            'Goalkeeper': [
                'status_numeric', 'playing_probability', 'saves_per_90', 'conceded_per_90',
                'clean_sheet_prob', 'minutes_consistency_3', 'team_strength_composite',
                'rolling_points_3', 'rolling_saves_3', 'rolling_conceded_3',
                'form_adjusted', 'form_score', 'save_percentage', 'gk_bonus_per_90',
                'played_60', 'played_90', 'got_bonus'
            ],
            'Defender': [
                'status_numeric', 'playing_probability', 'clean_sheet_prob', 'goals_conceded',
                'def_actions_per_90', 'tackles_per_90', 'team_strength_composite',
                'rolling_points_3', 'rolling_points_5', 'minutes_consistency_3',
                'form_adjusted', 'form_score', 'def_clean_sheet_prob',
                'played_60', 'played_90', 'got_bonus', 'bonus'
            ],
            'Midfielder': [
                'status_numeric', 'playing_probability', 'expected_assists', 'expected_goals',
                'xa_per_90', 'goal_involvement_per_90', 'team_strength_composite',
                'rolling_points_3', 'rolling_points_5', 'rolling_xa_3',
                'form_adjusted', 'form_score', 'creativity_threat_ratio',
                'played_60', 'played_90', 'got_bonus', 'influence_current'
            ],
            'Forward': [
                'status_numeric', 'playing_probability', 'expected_goals', 'goals_scored',
                'xg_per_90', 'goal_efficiency', 'team_strength_composite',
                'rolling_points_3', 'rolling_points_5', 'rolling_xg_3',
                'form_adjusted', 'form_score', 'threat_per_minute',
                'played_60', 'played_90', 'got_bonus', 'threat_current'
            ]
        }
        
        # Start with position-specific priorities that exist in X
        priority_features = position_priorities.get(position, [])
        available_priorities = [f for f in priority_features if f in X.columns]
        
        print(f"    Found {len(available_priorities)}/{len(priority_features)} priority features")
        
        # If we have enough priority features, use them
        if len(available_priorities) >= 10:
            selected_features = available_priorities[:self.max_features]
            print(f"    Using {len(selected_features)} priority features")
            return selected_features, {'method': 'position_priorities'}
        
        # Otherwise, use hybrid selection
        return self.select_features_hybrid_position(X, y, position, available_priorities)
    
    def select_features_hybrid_position(self, X: pd.DataFrame, y: pd.Series,
                                       position: str, priority_features: List[str]) -> Tuple[List[str], Dict]:
        """
        Hybrid feature selection with position awareness
        
        Args:
            X: Feature DataFrame
            y: Target Series
            position: Player position
            priority_features: Position-specific priority features
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        selection_info = {
            'position': position,
            'priority_features': priority_features,
            'method': 'hybrid'
        }
        
        # Method 1: Correlation selection
        try:
            corr_features, corr_importance = self.select_features_simple(X, y, position)
            selection_info['corr_features'] = corr_features
            selection_info['corr_importance'] = corr_importance
        except Exception as e:
            print(f"    ⚠ Correlation selection failed: {e}")
            corr_features = []
        
        # Method 2: Mutual information
        try:
            mi_features = self.select_features_mutual_info(X, y)
            selection_info['mi_features'] = mi_features
        except Exception as e:
            print(f"    ⚠ Mutual information failed: {e}")
            mi_features = []
        
        # Combine all feature sources (priorities first)
        all_features = list(set(priority_features + corr_features + mi_features))
        
        if not all_features:
            print(f"    ⚠ No features selected, using all available features")
            all_features = X.columns.tolist()[:self.max_features]
        
        # Remove highly correlated features
        try:
            final_features = self.remove_highly_correlated(X[all_features], all_features)
        except:
            final_features = all_features[:self.max_features]
        
        # Ensure we don't exceed max features
        final_features = final_features[:self.max_features]
        
        selection_info['final_features'] = final_features
        selection_info['feature_count'] = len(final_features)
        
        print(f"    ✓ Selected {len(final_features)} features for {position}")
        
        return final_features, selection_info
    
    def select_features_simple(self, X: pd.DataFrame, y: pd.Series, 
                              position: str = None) -> Tuple[List[str], pd.DataFrame]:
        """
        Simple feature selection using correlation only
        
        Args:
            X: Feature DataFrame
            y: Target Series
            position: Player position (for logging)
            
        Returns:
            Tuple of (selected_features, importance_df)
        """
        try:
            # Calculate correlation with target
            corr_with_target = X.corrwith(y).abs()
            corr_with_target = corr_with_target.dropna().sort_values(ascending=False)
            
            # Create importance DataFrame
            importance = pd.DataFrame({
                'feature': corr_with_target.index,
                'importance': corr_with_target.values
            })
            
            # Select top features
            selected_features = importance.head(self.max_features)['feature'].tolist()
            
            if position:
                print(f"    Correlation: {len(selected_features)} features")
            
            return selected_features, importance
            
        except Exception as e:
            print(f"  ⚠ Simple feature selection failed: {e}")
            return [], pd.DataFrame()
    
    def select_features_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select features using mutual information
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        try:
            # Handle NaN values
            X_filled = X.fillna(0)
            
            # Calculate mutual information
            mi = mutual_info_regression(X_filled, y, random_state=42)
            
            # Create importance DataFrame
            mi_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mi
            })
            mi_df = mi_df.sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = mi_df.head(self.max_features)['feature'].tolist()
            
            return selected_features
        except Exception as e:
            print(f"  ⚠ Mutual information failed: {e}")
            return []
    
    def remove_highly_correlated(self, X: pd.DataFrame, features: List[str], 
                                threshold: float = 0.8) -> List[str]:
        """
        Remove highly correlated features
        
        Args:
            X: Feature DataFrame
            features: List of feature names
            threshold: Correlation threshold
            
        Returns:
            List of filtered features
        """
        if len(features) <= 1:
            return features
        
        try:
            correlation_matrix = X[features].corr().abs()
            selected_features = []
            
            for feature in features:
                if not selected_features:
                    selected_features.append(feature)
                else:
                    # Check correlation with already selected features
                    max_corr = correlation_matrix.loc[feature, selected_features].max()
                    if max_corr < threshold or pd.isna(max_corr):
                        selected_features.append(feature)
                
                if len(selected_features) >= self.max_features:
                    break
            
            return selected_features
        except:
            return features[:self.max_features]
    
    def select_features_for_raw_prediction(self, X: pd.DataFrame, y: pd.Series,
                                         position: str = None) -> Tuple[List[str], Dict]:
        """
        Specialized feature selection for RAW point prediction
        
        Args:
            X: Feature DataFrame
            y: Target Series
            position: Player position
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        # Use position-aware selection
        return self.select_features_position_aware(X, y, position)