"""
Analyze RAW point prediction models and feature importance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from ml_model.config import MODELS_DIR, FEATURE_CATEGORIES, MODEL_PREFIX_RAW


class ModelAnalyzer:
    def __init__(self, models_dir: str = MODELS_DIR, model_prefix: str = MODEL_PREFIX_RAW):
        """
        Initialize ModelAnalyzer for RAW point models
        
        Args:
            models_dir: Directory containing saved models
            model_prefix: Prefix for model files ('raw_' for raw point models)
        """
        self.models_dir = models_dir
        self.model_prefix = model_prefix
        self.feature_importances = {}
        
    def load_raw_feature_importances(self, position: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load feature importances from RAW point models
        
        Args:
            position: Specific position to load (None for all)
            
        Returns:
            Dictionary of feature importances by position
        """
        importances = {}
        
        positions = [position] if position else ['GKP', 'DEF', 'MID', 'FWD']
        
        for pos in positions:
            # Try to load raw feature importance file
            importance_file = os.path.join(self.models_dir, f'{self.model_prefix}{pos}_feature_importance.pkl')
            
            if os.path.exists(importance_file):
                with open(importance_file, 'rb') as f:
                    importances[pos] = pickle.load(f)
            else:
                # Try to extract from model
                model_file = os.path.join(self.models_dir, f'{self.model_prefix}{pos}_lgb_model.txt')
                if os.path.exists(model_file):
                    import lightgbm as lgb
                    model = lgb.Booster(model_file=model_file)
                    
                    # Get feature names from features file
                    features_file = os.path.join(self.models_dir, f'{self.model_prefix}{pos}_features.pkl')
                    if os.path.exists(features_file):
                        with open(features_file, 'rb') as f:
                            feature_names = pickle.load(f)
                        
                        # Get feature importance
                        importance_values = model.feature_importance(importance_type='gain')
                        
                        # Create importance DataFrame
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importance_values
                        })
                        importance_df = importance_df.sort_values('importance', ascending=False)
                        importances[pos] = importance_df
        
        self.feature_importances = importances
        return importances
    
    def analyze_raw_feature_importance(self, position: str = None) -> pd.DataFrame:
        """
        Analyze feature importance across positions for RAW point prediction
        
        Args:
            position: Specific position to analyze (None for all)
            
        Returns:
            DataFrame with feature importance analysis
        """
        if not self.feature_importances:
            self.load_raw_feature_importances(position)
        
        if position and position in self.feature_importances:
            return self.feature_importances[position]
        
        # Combine importance across positions
        all_features = set()
        for pos, importance_df in self.feature_importances.items():
            if importance_df is not None and not importance_df.empty:
                all_features.update(importance_df['feature'].tolist())
        
        analysis_df = pd.DataFrame(index=list(all_features))
        
        for pos, importance_df in self.feature_importances.items():
            if importance_df is not None and not importance_df.empty:
                # Normalize importance
                importance_df['importance_normalized'] = (
                    importance_df['importance'] / importance_df['importance'].sum()
                )
                
                # Merge with analysis_df
                for _, row in importance_df.iterrows():
                    analysis_df.loc[row['feature'], f'{pos}_importance'] = row['importance_normalized']
        
        # Fill NaN with 0
        analysis_df = analysis_df.fillna(0)
        
        # Calculate statistics
        analysis_df['mean_importance'] = analysis_df.mean(axis=1)
        analysis_df['std_importance'] = analysis_df.std(axis=1)
        analysis_df['max_importance'] = analysis_df.max(axis=1)
        analysis_df['count_positions'] = (analysis_df.iloc[:, :4] > 0).sum(axis=1)
        
        # Sort by mean importance
        analysis_df = analysis_df.sort_values('mean_importance', ascending=False)
        
        return analysis_df
    
    def categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features based on predefined categories for RAW point prediction
        
        Args:
            features: List of feature names
            
        Returns:
            Dictionary mapping categories to feature lists
        """
        categorized = {category: [] for category in FEATURE_CATEGORIES.keys()}
        categorized['Other'] = []
        
        for feature in features:
            assigned = False
            for category, patterns in FEATURE_CATEGORIES.items():
                for pattern in patterns:
                    if pattern in feature:
                        categorized[category].append(feature)
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                categorized['Other'].append(feature)
        
        return categorized
    
    def plot_raw_feature_importance(self, position: str = None, top_n: int = 20,
                                   save_path: str = None):
        """
        Plot feature importance for RAW point prediction
        
        Args:
            position: Position to plot (None for overall)
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        if position and position in self.feature_importances:
            # Plot for specific position
            importance_df = self.feature_importances[position]
            if importance_df is not None and not importance_df.empty:
                self._plot_single_position_raw(importance_df, position, top_n, save_path)
        else:
            # Plot overall importance
            analysis_df = self.analyze_raw_feature_importance()
            self._plot_overall_importance_raw(analysis_df, top_n, save_path)
    
    def _plot_single_position_raw(self, importance_df: pd.DataFrame, 
                                 position: str, top_n: int, save_path: str):
        """Plot feature importance for a single position (RAW points)"""
        # Get top features
        top_features = importance_df.head(top_n).copy()
        
        # Categorize features
        categories = self.categorize_features(top_features['feature'].tolist())
        
        # Assign colors by category
        category_colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        color_map = {}
        for (category, features), color in zip(categories.items(), category_colors):
            for feature in features:
                if feature in top_features['feature'].values:
                    color_map[feature] = color
        
        top_features['color'] = top_features['feature'].map(color_map)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                      color=top_features['color'].values)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title(f'Top {top_n} Feature Importance for {position} (RAW Points)')
        
        # Add category legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map.get(feature, 'gray'), 
                               label=category)
                          for category, features in categories.items()
                          for feature in features[:1]]  # One per category
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  RAW feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_overall_importance_raw(self, analysis_df: pd.DataFrame, 
                                    top_n: int, save_path: str):
        """Plot overall feature importance across positions (RAW points)"""
        # Get top features by mean importance
        top_features = analysis_df.head(top_n).copy()
        
        # Prepare data for grouped bar chart
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        feature_names = top_features.index.tolist()
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 10))
        
        x = np.arange(len(feature_names))
        width = 0.2
        
        for i, pos in enumerate(positions):
            if f'{pos}_importance' in top_features.columns:
                importance_values = top_features[f'{pos}_importance'].fillna(0).values
                offset = (i - 1.5) * width
                ax.bar(x + offset, importance_values, width, label=pos)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Normalized Importance')
        ax.set_title(f'Top {top_n} Feature Importance by Position (RAW Points)')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Overall RAW feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def generate_raw_model_report(self) -> Dict[str, any]:
        """
        Generate comprehensive model analysis report for RAW point prediction
        
        Returns:
            Dictionary with model analysis
        """
        report = {
            'model_files': [],
            'feature_importance_summary': {},
            'top_features_by_position': {},
            'feature_category_distribution': {},
            'model_type': 'raw_points'
        }
        
        # List raw model files
        model_files = [f for f in os.listdir(self.models_dir) if f.startswith(self.model_prefix) and f.endswith(('.txt', '.json', '.pkl'))]
        report['model_files'] = model_files
        
        # Load feature importances
        self.load_raw_feature_importances()
        
        # Analyze each position
        for position in ['GKP', 'DEF', 'MID', 'FWD']:
            if position in self.feature_importances:
                importance_df = self.feature_importances[position]
                
                if importance_df is not None and not importance_df.empty:
                    # Top features
                    top_features = importance_df.head(10)
                    report['top_features_by_position'][position] = top_features[['feature', 'importance']].to_dict('records')
                    
                    # Feature categories
                    categories = self.categorize_features(importance_df['feature'].tolist())
                    category_counts = {cat: len(feats) for cat, feats in categories.items()}
                    report['feature_category_distribution'][position] = category_counts
        
        # Overall statistics
        if self.feature_importances:
            analysis_df = self.analyze_raw_feature_importance()
            
            # Top overall features
            top_overall = analysis_df.head(10)
            report['top_overall_features'] = top_overall[['mean_importance', 'std_importance']].to_dict()
            
            # Feature category distribution overall
            all_features = analysis_df.index.tolist()
            all_categories = self.categorize_features(all_features)
            report['overall_category_distribution'] = {cat: len(feats) for cat, feats in all_categories.items()}
        
        return report
    
    def compare_feature_categories_raw(self) -> pd.DataFrame:
        """
        Compare feature importance across categories for RAW point prediction
        
        Returns:
            DataFrame with category comparison
        """
        if not self.feature_importances:
            self.load_raw_feature_importances()
        
        categories = list(FEATURE_CATEGORIES.keys()) + ['Other']
        category_data = {category: [] for category in categories}
        
        for position, importance_df in self.feature_importances.items():
            if importance_df is not None and not importance_df.empty:
                # Normalize importance
                importance_df['importance_normalized'] = (
                    importance_df['importance'] / importance_df['importance'].sum()
                )
                
                # Categorize features
                categories_map = self.categorize_features(importance_df['feature'].tolist())
                
                for category, features in categories_map.items():
                    if features:
                        category_importance = importance_df[
                            importance_df['feature'].isin(features)
                        ]['importance_normalized'].sum()
                        category_data[category].append(category_importance)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            category: {
                'mean': np.mean(values) if values else 0,
                'std': np.std(values) if values else 0,
                'min': np.min(values) if values else 0,
                'max': np.max(values) if values else 0,
                'count': len(values)
            }
            for category, values in category_data.items()
        }).T
        
        comparison_df = comparison_df.sort_values('mean', ascending=False)
        
        return comparison_df
    
    def analyze_performance_features(self) -> Dict[str, List[str]]:
        """
        Analyze which performance features are most important for RAW point prediction
        
        Returns:
            Dictionary with top performance features by category
        """
        if not self.feature_importances:
            self.load_raw_feature_importances()
        
        performance_features = {}
        
        for position, importance_df in self.feature_importances.items():
            if importance_df is not None and not importance_df.empty:
                # Get top 10 features
                top_features = importance_df.head(10)['feature'].tolist()
                
                # Categorize them
                categories = self.categorize_features(top_features)
                
                # Store by position
                performance_features[position] = {}
                for category, features in categories.items():
                    if features:
                        performance_features[position][category] = features
        
        return performance_features