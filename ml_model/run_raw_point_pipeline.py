"""
Main pipeline for RAW point prediction (Layer 1 of separated approach)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import pickle
import sqlite3
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

from ml_model.data_loader import DataLoader
from ml_model.feature_selector import FeatureSelector
from ml_model.model_trainer import ModelTrainer
from ml_model.model_evaluator import ModelEvaluator
from ml_model.model_analyzer import ModelAnalyzer
from ml_model.raw_prediction_generator import RawPredictionGenerator
from ml_model.config import MODELS_DIR, PREDICTIONS_DIR, DATABASE_PATH, POSITIONS, MODEL_PREFIX_RAW


def train_raw_points_models(retrain: bool = True, validate: bool = True):
    """
    Train models to predict RAW POINTS (no value metrics)
    
    Returns:
        bool: Success status
    """
    print("\n" + "="*70)
    print("FPL SIMULATOR - RAW POINT PREDICTION MODEL TRAINING")
    print("EXCLUDING all cost/value metrics for pure performance prediction")
    print("="*70 + "\n")
    
    # Initialize components
    data_loader = DataLoader()
    
    # Get latest gameweek info
    latest_gw = data_loader.get_latest_gameweek()
    available_gws = data_loader.get_available_gameweeks()
    
    print(f" DATABASE STATUS:")
    print(f"  Latest available gameweek: {latest_gw}")
    print(f"  Available gameweeks: {available_gws}")
    print(f"  Next gameweek to predict: {latest_gw + 1}")
    
    # Step 1: Load training data for RAW point prediction
    print("\n" + "="*60)
    print("STEP 1: LOADING TRAINING DATA FOR RAW POINT PREDICTION")
    print("="*60)
    
    try:
        training_data = data_loader.load_raw_training_data(exclude_value_features=True)
        
        if training_data.empty:
            print("  ✗ No training data found")
            return False
        
        print(f"  ✓ Loaded {len(training_data)} samples")
        
        # Check feature count
        feature_cols = [col for col in training_data.columns if col not in [
            'player_id', 'season', 'feature_generation_time', 'target_gameweek',
            'actual_points', 'web_name', 'team_id'
        ]]
        
        print(f"  ✓ {len(feature_cols)} features available for raw point prediction")
        
        # Check for critical performance features
        performance_features = [
            'playing_probability', 'status_numeric', 'form_adjusted', 
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'rolling_points_3', 'rolling_points_5', 'form_score',
            'influence', 'creativity', 'threat', 'ict_index'
        ]
        
        present_features = [f for f in performance_features if f in training_data.columns]
        print(f"  ✓ Performance features present: {len(present_features)}/{len(performance_features)}")
        
        if len(present_features) < 8:
            print(f"  ⚠ Warning: May need more performance features")
        
    except Exception as e:
        print(f"  ✗ Failed to load training data: {e}")
        return False
    
    # Step 2: Check if raw models already exist
    print("\n" + "="*60)
    print("STEP 2: CHECKING FOR EXISTING RAW POINT MODELS")
    print("="*60)
    
    models_exist = all([
        os.path.exists(os.path.join(MODELS_DIR, f'{MODEL_PREFIX_RAW}{pos}_lgb_model.txt'))
        for pos in ['GKP', 'DEF', 'MID', 'FWD']
    ])
    
    if retrain or not models_exist:
        print(" Training/retraining RAW point models...")
        
        # Split by position
        position_data = {}
        for position in POSITIONS:
            pos_data = training_data[training_data['position'] == position].copy()
            if len(pos_data) > 50:
                position_data[position] = pos_data
                print(f"  {position}: {len(pos_data)} samples")
        
        if len(position_data) < 2:
            print("  ✗ Not enough position data for training")
            return False
        
        # Initialize trainer and selector
        trainer = ModelTrainer(model_type='lightgbm', model_prefix=MODEL_PREFIX_RAW)
        selector = FeatureSelector()
        
        # Feature selection and model training for each position
        selected_features = {}
        position_training_data = {}
        
        for position, pos_data in position_data.items():
            print(f"\n  Processing {position}...")
            
            # Prepare raw features and target
            X, y = data_loader.prepare_raw_features_target(pos_data)
            
            if len(X) < 20:
                print(f"    ⚠ Skipping {position} - insufficient data")
                continue
            
            # Feature selection for raw point prediction
            selected_features[position], selection_info = selector.select_features_for_raw_prediction(
                X, y, position
            )
            
            print(f"    Selected {len(selected_features[position])} features")
            
            # Store training data
            position_training_data[position] = (X[selected_features[position]], y)
        
        # Train models
        if position_training_data:
            trained_models = trainer.train_position_models_raw(position_training_data, selected_features)
            
            if trained_models:
                # Save models
                trainer.save_models()
                
                print(f"\n  ✓ Successfully trained and saved {len(trained_models)} RAW point models")
                
                # Generate model summary
                summary = trainer.get_model_summary()
                print(f"  Model Summary:")
                print(f"    Total samples: {summary['total_samples']}")
                print(f"    Total features: {summary['total_features']}")
                for position, info in summary['positions'].items():
                    print(f"    {position}: {info['n_samples']} samples, {info['n_features']} features")
            else:
                print("  ✗ No models were trained")
                return False
        else:
            print("  ✗ No position data available for training")
            return False
    else:
        print("  Using existing RAW point models")
    
    # Step 3: Validation (optional)
    if validate and 'target_gameweek' in training_data.columns:
        print("\n" + "="*60)
        print("STEP 3: VALIDATING RAW POINT MODELS")
        print("="*60)
        
        evaluator = ModelEvaluator()
        
        # Get available gameweeks for validation
        available_training_gws = sorted(training_data['target_gameweek'].unique())
        
        if len(available_training_gws) >= 4:
            print("  Running walk-forward validation...")
            
            validation_results = []
            
            # Use last 3 gameweeks for validation
            for test_gw in available_training_gws[-3:]:
                if test_gw <= 2:  # Skip early gameweeks
                    continue
                    
                # Prepare split
                train_data = training_data[training_data['target_gameweek'] < test_gw].copy()
                test_data = training_data[training_data['target_gameweek'] == test_gw].copy()
                
                if len(train_data) > 100 and len(test_data) > 20:
                    # Train temporary model for validation
                    temp_trainer = ModelTrainer(model_type='lightgbm', model_prefix='temp_')
                    temp_selector = FeatureSelector()
                    
                    # Split by position
                    train_by_position = data_loader.split_by_position(train_data)
                    test_by_position = data_loader.split_by_position(test_data)
                    
                    temp_selected_features = {}
                    temp_training_data = {}
                    
                    for position, pos_train in train_by_position.items():
                        if len(pos_train) < 20:
                            continue
                            
                        X_train, y_train = data_loader.prepare_raw_features_target(pos_train)
                        features, _ = temp_selector.select_features_for_raw_prediction(X_train, y_train, position)
                        temp_selected_features[position] = features
                        temp_training_data[position] = (X_train[features], y_train)
                    
                    # Train temporary models
                    temp_models = temp_trainer.train_position_models_raw(temp_training_data, temp_selected_features)
                    
                    # Evaluate on test data
                    for position, pos_test in test_by_position.items():
                        if position not in temp_models or len(pos_test) < 5:
                            continue
                            
                        X_test, y_test = data_loader.prepare_raw_features_target(pos_test)
                        model_info = temp_models[position]
                        features = model_info['features']
                        
                        # Ensure features exist
                        test_features = [f for f in features if f in X_test.columns]
                        if not test_features:
                            continue
                            
                        X_test_selected = X_test[test_features]
                        model = model_info['model']
                        
                        # Make predictions
                        y_pred = model.predict(X_test_selected, num_iteration=model.best_iteration)
                        
                        # Calculate metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        # R² calculation
                        ss_res = np.sum((y_test - y_pred) ** 2)
                        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        validation_results.append({
                            'test_gw': test_gw,
                            'position': position,
                            'n_samples': len(y_test),
                            'mae': mae,
                            'r2': r2
                        })
            
            if validation_results:
                val_df = pd.DataFrame(validation_results)
                
                print("\n  ENHANCED VALIDATION REPORT:")
                print("  " + "-" * 60)
                
            
                # Summary by position
                for position in POSITIONS:
                    pos_results = val_df[val_df['position'] == position]
                    if not pos_results.empty:
                        avg_mae = pos_results['mae'].mean()
                        mae_std = pos_results['mae'].std()
                        avg_r2 = pos_results['r2'].mean()
                        
                        mae_ci = 1.96 * mae_std / np.sqrt(len(pos_results))
                        
                        # Performance rating
                        if avg_mae < 1.2:
                            rating = "EXCELLENT"
                        elif avg_mae < 1.5:
                            rating = "GOOD"
                        elif avg_mae < 1.8:
                            rating = "ACCEPTABLE"
                        else:
                            rating = "NEEDS IMPROVEMENT"
                        
                        print(f"   {position}:")
                        print(f"    MAE: {avg_mae:.3f} ± {mae_ci:.3f} ({rating})")
                        print(f"    R²:  {avg_r2:.3f}")
                        print(f"    GWs tested: {pos_results['test_gw'].tolist()}")
                        
                # Overall metrics
                overall_mae = val_df['mae'].mean()
                overall_r2 = val_df['r2'].mean()
                
                print("\n  OVERALL PERFORMANCE:")
                print(f"   Mean MAE: {overall_mae:.3f} ({'PASS' if overall_mae < 1.5 else 'WARNING'})")
                print(f"   Mean R²: {overall_r2:.3f} ({'PASS' if overall_r2 > 0.3 else 'WARNING'})")
                
               
        else:
            print("  Not enough gameweeks for validation (need at least 4)")
    else:
        print("\n  Skipping validation")
    
    # Step 4: Model analysis
    print("\n" + "="*60)
    print("STEP 4: RAW POINT MODEL ANALYSIS")
    print("="*60)
    
    try:
        analyzer = ModelAnalyzer(model_prefix=MODEL_PREFIX_RAW)
        report = analyzer.generate_raw_model_report()
        
        if report['model_files']:
            print(f"  Model files found: {len(report['model_files'])}")
            
            # Feature importance analysis
            importances = analyzer.load_raw_feature_importances()
            if importances:
                print("\n  TOP FEATURES BY POSITION (RAW POINTS):")
                for position in ['GKP', 'DEF', 'MID', 'FWD']:
                    if position in importances and importances[position] is not None:
                        top_features = importances[position].head(3)['feature'].tolist()
                        print(f"   {position}: {', '.join(top_features)}")
                
                # Save feature analysis
                analysis_file = os.path.join(PREDICTIONS_DIR, f'raw_feature_analysis_{datetime.now().strftime("%Y%m%d")}.json')
                import json
                with open(analysis_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"\n  Feature analysis saved to: {analysis_file}")
    except Exception as e:
        print(f"  ⚠ Model analysis skipped: {e}")
    
    print("\n" + "="*70)
    print("RAW POINT MODEL TRAINING COMPLETED!")
    print("="*70)
    
    return True


def generate_raw_predictions(next_gw: int = None):
    """
    Generate RAW point predictions using trained models
    
    Args:
        next_gw: Specific gameweek to predict (None for next)
        
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*70)
    print("GENERATING RAW POINT PREDICTIONS")
    print("="*70 + "\n")
    
    # Initialize prediction generator
    predictor = RawPredictionGenerator(model_type='lightgbm')
    
    # Generate predictions
    if next_gw is None:
        next_gw = predictor.get_next_gameweek()
    
    print(f"  Predicting GW{next_gw}...")
    predictions = predictor.predict_next_gameweek(next_gw)
    
    if predictions.empty:
        print("  ✗ No predictions generated")
        return pd.DataFrame()
    
    print(f"\n  ✓ Generated {len(predictions)} RAW point predictions")
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = os.path.join(PREDICTIONS_DIR, f'raw_points_predictions_gw{next_gw}.csv')
    predictions.to_csv(raw_file, index=False)
    
    latest_file = os.path.join(PREDICTIONS_DIR, f'latest_raw_points_predictions_gw{next_gw}.csv')
    predictions.to_csv(latest_file, index=False)
    
    print(f"  ✓ Predictions saved to: {raw_file}")
    print(f"  ✓ Latest copy: {latest_file}")
    
    return predictions


def run_complete_raw_pipeline(retrain: bool = True, validate: bool = True, 
                            generate_predictions: bool = True):
    """
    Complete pipeline for RAW point prediction
    
    Args:
        retrain: Whether to retrain models
        validate: Whether to run validation
        generate_predictions: Whether to generate predictions
        
    Returns:
        bool: Success status
    """
    print("\n" + "="*70)
    print("FPL SIMULATOR - COMPLETE RAW POINT PREDICTION PIPELINE")
    print("Layer 1: Pure performance prediction (no cost/value features)")
    print("="*70 + "\n")
    
    # Step 1: Train raw point models
    training_success = train_raw_points_models(retrain=retrain, validate=validate)
    
    if not training_success:
        print("  ✗ Raw point model training failed")
        return False
    
    # Step 2: Generate predictions
    if generate_predictions:
        predictions = generate_raw_predictions()
        
        if predictions.empty:
            print("  ✗ Raw point prediction generation failed")
            return False
        
        # Analyze predictions
        print("\n" + "="*60)
        print("PREDICTION ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print(f"  Prediction Statistics:")
        print(f"    Total players predicted: {len(predictions)}")
        print(f"    Average predicted points: {predictions['predicted_points'].mean():.2f}")
        print(f"    Range: {predictions['predicted_points'].min():.1f} - {predictions['predicted_points'].max():.1f}")
        
        if 'now_cost' in predictions.columns:
            # Add value metrics for optimization phase
            predictions['now_cost_millions'] = predictions['now_cost'] / 10
            predictions['points_per_million'] = predictions.apply(
                lambda row: row['predicted_points'] / row['now_cost_millions'] 
                if row['now_cost_millions'] > 0 else 0,
                axis=1
            )
            
            print(f"    Average cost: £{predictions['now_cost_millions'].mean():.1f}M")
            print(f"    Average value: {predictions['points_per_million'].mean():.2f} pts/£M")
            
            
        
        # Position breakdown
        print(f"\n  Position Breakdown:")
        for position in POSITIONS:
            pos_pred = predictions[predictions['position'] == position]
            if not pos_pred.empty:
                print(f"    {position}: {len(pos_pred)} players")
                print(f"      Avg points: {pos_pred['predicted_points'].mean():.2f}")
                if 'now_cost_millions' in pos_pred.columns:
                    print(f"      Avg cost: £{pos_pred['now_cost_millions'].mean():.1f}M")
                    print(f"      Avg value: {pos_pred['points_per_million'].mean():.2f} pts/£M")
    
    print("\n" + "="*70)
    print("RAW POINT PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: Run optimization with raw point predictions")
    print("Command: python optimization/run_optimization.py")
    
    return True



def main():
    """Main entry point for raw point prediction pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FPL Raw Point Prediction Pipeline')
    parser.add_argument('--retrain', action='store_true', default=True, 
                       help='Retrain raw point models')
    parser.add_argument('--no-validate', action='store_true', 
                       help='Skip validation')
    parser.add_argument('--predict-only', action='store_true', 
                       help='Only generate predictions (skip training)')
    parser.add_argument('--train-only', action='store_true', 
                       help='Only train models (skip prediction)')
    parser.add_argument('--gameweek', type=int, 
                       help='Specific gameweek to predict')
    parser.add_argument('--validate-gw', type=int, 
                       help='Validate predictions for specific gameweek')
    
    args = parser.parse_args()
    
    if args.validate_gw:
        # Validate specific gameweek
        print(f"Validating raw predictions for GW{args.validate_gw}...")
        predictor = RawPredictionGenerator()
        validation_df = predictor.validate_raw_predictions(args.validate_gw)
        return not validation_df.empty
    
    elif args.predict_only:
        # Only generate predictions
        predictions = generate_raw_predictions(args.gameweek)
        return not predictions.empty
    
    elif args.train_only:
        # Only train models
        return train_raw_points_models(retrain=args.retrain, validate=not args.no_validate)
    
    else:
        # Run complete pipeline
        return run_complete_raw_pipeline(
            retrain=args.retrain,
            validate=not args.no_validate,
            generate_predictions=not args.train_only
        )


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)