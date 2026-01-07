"""
Evaluate RAW point prediction models with walk-forward validation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ml_model.config import MAE_THRESHOLD, R2_THRESHOLD, POSITIONS


class ModelEvaluator:
    def __init__(self):
        """Initialize ModelEvaluator for RAW point prediction"""
        self.evaluation_history = []
        self.feature_importance_history = []
    
    def walk_forward_validation(self, data: pd.DataFrame, 
                                trainer, 
                                selector,
                                start_gw: int = 2,
                                end_gw: int = 6) -> pd.DataFrame:
        """
        Perform walk-forward validation (time-series cross-validation)
        for RAW point prediction
        
        Args:
            data: Complete training data
            trainer: ModelTrainer instance
            selector: FeatureSelector instance
            start_gw: Starting gameweek
            end_gw: Ending gameweek
            
        Returns:
            DataFrame with validation results
        """
        from ml_model.data_loader import DataLoader
        
        data_loader = DataLoader()
        all_results = []
        
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION FOR RAW POINT PREDICTION")
        print("="*60)
        
        # Walk-forward: train on GW2-3, test on GW4, then train on GW2-4, test on GW5, etc.
        for test_gw in range(4, end_gw + 1):
            print(f"\nTest Gameweek {test_gw}:")
            print(f"  Training on GW{start_gw}-{test_gw-1}")
            print(f"  Testing on GW{test_gw}")
            
            # Prepare time-series split
            train_data, test_data = data_loader.prepare_time_series_split(
                data, train_gw_end=test_gw-1, test_gw=test_gw
            )
            
            # Skip if not enough data
            if len(train_data) < 50 or len(test_data) < 10:
                print(f"  Skipping - insufficient data (train: {len(train_data)}, test: {len(test_data)})")
                continue
            
            # Split by position for training
            train_by_position = data_loader.split_by_position(train_data)
            test_by_position = data_loader.split_by_position(test_data)
            
            # Feature selection and model training for each position
            position_models = {}
            selected_features = {}
            
            for position, pos_train_data in train_by_position.items():
                if len(pos_train_data) < 10:
                    continue
                
                # Prepare raw features and target (exclude cost/value)
                X_train, y_train = data_loader.prepare_raw_features_target(pos_train_data)
                
                # Feature selection for raw point prediction
                selected_features[position], selection_info = selector.select_features_for_raw_prediction(
                    X_train, y_train, position
                )
                
                # Train model for raw point prediction
                trainer.position_models = {}  # Reset for each fold
                position_data = {position: (X_train[selected_features[position]], y_train)}
                position_models.update(trainer.train_position_models_raw(position_data, selected_features))
            
            # Evaluate on test data
            test_results = []
            
            for position, pos_test_data in test_by_position.items():
                if position not in position_models or len(pos_test_data) < 5:
                    continue
                
                # Prepare test features
                X_test, y_test = data_loader.prepare_raw_features_target(pos_test_data)
                model_info = position_models[position]
                
                # Ensure test features match training features
                test_features = [f for f in selected_features[position] if f in X_test.columns]
                if not test_features:
                    continue
                
                X_test_selected = X_test[test_features]
                
                # Make RAW point predictions
                model = model_info['model']
                
                if trainer.model_type == 'lightgbm':
                    y_pred = model.predict(X_test_selected, num_iteration=model.best_iteration)
                elif trainer.model_type == 'xgboost':
                    import xgboost as xgb
                    dtest = xgb.DMatrix(X_test_selected)
                    y_pred = model.predict(dtest)
                else:
                    y_pred = model.predict(X_test_selected)
                
                # Calculate metrics for RAW point prediction
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                within_2 = np.mean(np.abs(y_test - y_pred) <= 2)
                
                test_results.append({
                    'test_gw': test_gw,
                    'position': position,
                    'n_samples': len(y_test),
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'within_2_points': within_2,
                    'actual_mean': y_test.mean(),
                    'predicted_mean': y_pred.mean(),
                    'prediction_type': 'raw_points'
                })
            
            # Store results for this fold
            if test_results:
                fold_results = pd.DataFrame(test_results)
                all_results.append(fold_results)
                
                # Print fold summary
                print(f"  Test Results for GW{test_gw} (RAW points):")
                for _, row in fold_results.iterrows():
                    status = "✓" if row['mae'] <= MAE_THRESHOLD else "⚠"
                    print(f"    {row['position']}: MAE={row['mae']:.2f}, R²={row['r2']:.3f} {status}")
        
        # Combine all results
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            self.evaluation_history = final_results
            
            # Calculate overall metrics
            print("\n" + "="*60)
            print("OVERALL WALK-FORWARD VALIDATION RESULTS (RAW POINTS)")
            print("="*60)
            
            overall_mae = final_results['mae'].mean()
            overall_r2 = final_results['r2'].mean()
            overall_within_2 = final_results['within_2_points'].mean()
            
            print(f"\nOverall Performance:")
            print(f"  Mean MAE: {overall_mae:.3f} {'✓' if overall_mae <= MAE_THRESHOLD else '✗'} (target: {MAE_THRESHOLD})")
            print(f"  Mean R²:  {overall_r2:.3f} {'✓' if overall_r2 >= R2_THRESHOLD else '✗'} (target: {R2_THRESHOLD})")
            print(f"  Within 2 points: {overall_within_2:.1%}")
            
            # Print by position
            print(f"\nBy Position:")
            for position in POSITIONS:
                pos_results = final_results[final_results['position'] == position]
                if not pos_results.empty:
                    pos_mae = pos_results['mae'].mean()
                    pos_r2 = pos_results['r2'].mean()
                    print(f"  {position}: MAE={pos_mae:.3f}, R²={pos_r2:.3f}")
            
            return final_results
        
        return pd.DataFrame()
    
    def evaluate_raw_predictions(self, actual_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate RAW point predictions against actual points
        
        Args:
            actual_df: DataFrame with actual points
            pred_df: DataFrame with predicted points
            
        Returns:
            DataFrame with evaluation metrics
        """
        if actual_df.empty or pred_df.empty:
            return pd.DataFrame()
        
        # Merge actual and predicted data
        merged_df = pred_df.merge(actual_df, on='player_id', how='inner')
        
        if merged_df.empty:
            print("  No matching players between predictions and actuals")
            return pd.DataFrame()
        
        evaluation_results = []
        
        # Evaluate by position
        for position in POSITIONS:
            pos_data = merged_df[merged_df['position'] == position]
            
            if len(pos_data) < 5:
                continue
            
            # Extract actual and predicted points
            y_actual = pos_data['actual_points']
            y_pred = pos_data['predicted_points']
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            within_2 = np.mean(np.abs(y_actual - y_pred) <= 2)
            
            evaluation_results.append({
                'position': position,
                'n_samples': len(y_actual),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'within_2_points': within_2,
                'actual_mean': y_actual.mean(),
                'predicted_mean': y_pred.mean(),
                'prediction_type': 'raw_points'
            })
        
        if evaluation_results:
            results_df = pd.DataFrame(evaluation_results)
            
            print(f"\nRAW Point Prediction Evaluation:")
            print(f"  Total samples: {len(merged_df)}")
            print(f"  Overall MAE: {results_df['mae'].mean():.3f}")
            print(f"  Overall R²: {results_df['r2'].mean():.3f}")
            
            return results_df
        
        return pd.DataFrame()
    
    def plot_evaluation_results(self, results: pd.DataFrame, save_path: str = None):
        """
        Plot evaluation results for RAW point prediction
        
        Args:
            results: DataFrame with evaluation results
            save_path: Path to save the plot
        """
        if results.empty:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. MAE by position and gameweek
        ax = axes[0]
        for position in results['position'].unique():
            pos_data = results[results['position'] == position]
            ax.plot(pos_data['test_gw'], pos_data['mae'], 'o-', label=position, alpha=0.7)
        ax.axhline(y=MAE_THRESHOLD, color='r', linestyle='--', alpha=0.5, label=f'Target ({MAE_THRESHOLD})')
        ax.set_xlabel('Test Gameweek')
        ax.set_ylabel('MAE')
        ax.set_title('MAE by Position and Gameweek (RAW Points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. R² by position and gameweek
        ax = axes[1]
        for position in results['position'].unique():
            pos_data = results[results['position'] == position]
            ax.plot(pos_data['test_gw'], pos_data['r2'], 'o-', label=position, alpha=0.7)
        ax.axhline(y=R2_THRESHOLD, color='r', linestyle='--', alpha=0.5, label=f'Target ({R2_THRESHOLD})')
        ax.set_xlabel('Test Gameweek')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score by Position and Gameweek')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted scatter
        ax = axes[2]
        # Need to recreate predictions for this plot
        ax.scatter(results['actual_mean'], results['predicted_mean'], alpha=0.6)
        ax.plot([results['actual_mean'].min(), results['actual_mean'].max()],
                [results['actual_mean'].min(), results['actual_mean'].max()],
                'r--', alpha=0.5, label='Perfect prediction')
        ax.set_xlabel('Actual Mean Points')
        ax.set_ylabel('Predicted Mean Points')
        ax.set_title('Actual vs Predicted (Position Means)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error distribution
        ax = axes[3]
        errors = []
        for position in results['position'].unique():
            pos_data = results[results['position'] == position]
            errors.extend(pos_data['mae'].values)
        ax.hist(errors, bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        ax.set_xlabel('MAE')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution (RAW Points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Samples per position
        ax = axes[4]
        position_counts = results.groupby('position')['n_samples'].sum()
        ax.bar(position_counts.index, position_counts.values, alpha=0.7)
        ax.set_xlabel('Position')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Samples per Position')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Within 2 points accuracy
        ax = axes[5]
        accuracy_by_pos = results.groupby('position')['within_2_points'].mean()
        ax.bar(accuracy_by_pos.index, accuracy_by_pos.values * 100, alpha=0.7)
        ax.set_xlabel('Position')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Predictions Within 2 Points (RAW)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  RAW point evaluation plots saved to: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, results: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive performance report for RAW point prediction
        
        Returns:
            Dictionary with performance metrics
        """
        if results.empty:
            return {}
        
        report = {
            'overall_metrics': {
                'mean_mae': results['mae'].mean(),
                'median_mae': results['mae'].median(),
                'mean_r2': results['r2'].mean(),
                'median_r2': results['r2'].median(),
                'mean_within_2': results['within_2_points'].mean(),
                'total_samples': results['n_samples'].sum(),
                'prediction_type': 'raw_points'
            },
            'by_position': {},
            'by_gameweek': {},
            'success_criteria': {
                'mae_target_met': results['mae'].mean() <= MAE_THRESHOLD,
                'r2_target_met': results['r2'].mean() >= R2_THRESHOLD,
                'mae_target': MAE_THRESHOLD,
                'r2_target': R2_THRESHOLD
            }
        }
        
        # By position
        for position in results['position'].unique():
            pos_data = results[results['position'] == position]
            report['by_position'][position] = {
                'mae': pos_data['mae'].mean(),
                'r2': pos_data['r2'].mean(),
                'within_2': pos_data['within_2_points'].mean(),
                'samples': pos_data['n_samples'].sum()
            }
        
        # By gameweek
        for gw in results['test_gw'].unique():
            gw_data = results[results['test_gw'] == gw]
            report['by_gameweek'][f'GW{gw}'] = {
                'mae': gw_data['mae'].mean(),
                'r2': gw_data['r2'].mean(),
                'samples': gw_data['n_samples'].sum()
            }
        
        return report
    
    def compare_value_vs_raw_predictions(self, raw_results: pd.DataFrame, 
                                       value_results: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare RAW point predictions vs value-based predictions
        
        Args:
            raw_results: Results from raw point prediction
            value_results: Results from value-based prediction (optional)
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        # Analyze raw point predictions
        if not raw_results.empty:
            for position in raw_results['position'].unique():
                pos_data = raw_results[raw_results['position'] == position]
                if not pos_data.empty:
                    comparison_data.append({
                        'position': position,
                        'prediction_type': 'raw_points',
                        'mae': pos_data['mae'].mean(),
                        'r2': pos_data['r2'].mean(),
                        'within_2': pos_data['within_2_points'].mean(),
                        'samples': pos_data['n_samples'].sum()
                    })
        
        # Analyze value-based predictions if available
        if value_results is not None and not value_results.empty:
            for position in value_results['position'].unique():
                pos_data = value_results[value_results['position'] == position]
                if not pos_data.empty:
                    comparison_data.append({
                        'position': position,
                        'prediction_type': 'value_based',
                        'mae': pos_data['mae'].mean(),
                        'r2': pos_data['r2'].mean(),
                        'within_2': pos_data['within_2_points'].mean(),
                        'samples': pos_data['n_samples'].sum()
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            print("\n" + "="*60)
            print("COMPARISON: RAW POINTS vs VALUE-BASED PREDICTIONS")
            print("="*60)
            
            # Group by prediction type
            for pred_type in comparison_df['prediction_type'].unique():
                type_data = comparison_df[comparison_df['prediction_type'] == pred_type]
                print(f"\n{pred_type.upper()} Prediction:")
                print(f"  Average MAE: {type_data['mae'].mean():.3f}")
                print(f"  Average R²: {type_data['r2'].mean():.3f}")
                print(f"  Within 2 points: {type_data['within_2'].mean():.1%}")
            
            return comparison_df
        
        return pd.DataFrame()