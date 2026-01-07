"""
Debug feature selection and usage
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
import numpy as np
from ml_model.config import MODELS_DIR

def analyze_feature_selection():
    """Analyze which features are being selected by models"""
    print("="*70)
    print("FEATURE SELECTION ANALYSIS")
    print("="*70)
    
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    
    for position in positions:
        # Load features file
        features_file = os.path.join(MODELS_DIR, f'{position}_features.pkl')
        
        if os.path.exists(features_file):
            with open(features_file, 'rb') as f:
                features = pickle.load(f)
            
            print(f"\n{position} Features ({len(features)}):")
            for i, feature in enumerate(features[:20]):  # Show first 20
                print(f"  {i+1:2}. {feature}")
            if len(features) > 20:
                print(f"  ... and {len(features)-20} more")
        
        # Load feature importance
        importance_file = os.path.join(MODELS_DIR, f'{position}_feature_importance.pkl')
        if os.path.exists(importance_file):
            with open(importance_file, 'rb') as f:
                importance_df = pickle.load(f)
            
            if importance_df is not None and not importance_df.empty:
                print(f"\n{position} Top 10 Feature Importances:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"  {i+1:2}. {row['feature']}: {row['importance']:.4f}")

def check_prediction_features():
    """Check what features are available in predictions"""
    print("\n" + "="*70)
    print("PREDICTION FEATURE ANALYSIS")
    print("="*70)
    
    # Load latest predictions
    preds_dir = '/Users/prabhuroka/Desktop/FPL_Project/data/predictions'
    pred_files = [f for f in os.listdir(preds_dir) if f.startswith('optimal_predictions_gw')]
    
    if pred_files:
        latest_pred = sorted(pred_files)[-1]
        pred_path = os.path.join(preds_dir, latest_pred)
        pred_df = pd.read_csv(pred_path)
        
        print(f"\nPredictions file: {latest_pred}")
        print(f"Total predictions: {len(pred_df)}")
        print(f"Features used: {pred_df['Features Used'].iloc[0] if 'Features Used' in pred_df.columns else 'Unknown'}")
        print(f"Total features available: {pred_df['Total Features Available'].iloc[0] if 'Total Features Available' in pred_df.columns else 'Unknown'}")
        
        # Sample player analysis
        print(f"\nSample player analysis (first 5):")
        for i, row in pred_df.head(5).iterrows():
            print(f"\nPlayer: {row.get('Web Name', 'Unknown')}")
            print(f"Position: {row.get('Position', 'Unknown')}")
            print(f"Predicted: {row.get('Predicted Points', 'Unknown')}")
            print(f"Features used: {row.get('Features Used', 'Unknown')}")

def check_training_data():
    """Check training data features"""
    print("\n" + "="*70)
    print("TRAINING DATA ANALYSIS")
    print("="*70)
    
    training_file = '/Users/prabhuroka/Desktop/FPL_Project/data/features/complete_training_data.csv'
    
    if os.path.exists(training_file):
        train_df = pd.read_csv(training_file)
        
        print(f"\nTraining data: {training_file}")
        print(f"Samples: {len(train_df)}")
        print(f"Features: {len(train_df.columns)}")
        
        # Categorize features
        categories = {
            'Basic Stats': [col for col in train_df.columns if col in [
                'minutes', 'total_points', 'goals_scored', 'assists', 
                'clean_sheets', 'goals_conceded', 'bonus', 'saves'
            ]],
            'Expected Stats': [col for col in train_df.columns if col.startswith('expected_') or 'xg' in col or 'xa' in col],
            'Form & Status': [col for col in train_df.columns if 'form' in col or 'status' in col or 'playing' in col],
            'Ownership': [col for col in train_df.columns if 'selected' in col or 'ownership' in col],
            'Value': [col for col in train_df.columns if 'cost' in col or 'value' in col],
            'Team': [col for col in train_df.columns if 'team' in col or 'strength' in col],
        }
        
        print("\nFeature Categories:")
        for category, features in categories.items():
            existing = [f for f in features if f in train_df.columns]
            if existing:
                print(f"  {category}: {len(existing)} features")
                if len(existing) <= 5:
                    print(f"    {existing}")
        
        # Missing critical features
        critical = ['selected_by_percent', 'form', 'playing_probability', 'status_numeric']
        missing = [f for f in critical if f not in train_df.columns]
        if missing:
            print(f"\n❌ MISSING CRITICAL FEATURES: {missing}")
        else:
            print(f"\n✓ All critical features present")

if __name__ == "__main__":
    analyze_feature_selection()
    check_prediction_features()
    check_training_data()