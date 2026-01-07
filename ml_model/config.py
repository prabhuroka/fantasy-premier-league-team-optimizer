"""
Configuration for ML models - COMPLETE VERSION with all required imports
"""
import os
from datetime import datetime

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')

# Database path
DATABASE_PATH = os.path.join(DATA_DIR, 'processed', 'sqlite', 'fpl_simulator.db')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ML Settings
RANDOM_SEED = 42
TEST_SIZE = 0.2  # For validation split (not used in walk-forward)
EARLY_STOPPING_ROUNDS = 50

# Model parameters (for RAW point prediction)
LIGHTGBM_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'random_state': RANDOM_SEED
}

# XGBoost parameters (if you want to use XGBoost as an alternative)
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.1,
    'lambda': 1.0,
    'n_jobs': -1,
    'random_state': RANDOM_SEED
}

# Feature selection
MAX_FEATURES_PER_MODEL = 30
MIN_FEATURE_IMPORTANCE = 0.01

# Position mapping
POSITIONS = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
POSITION_ABBREV = {
    'Goalkeeper': 'GKP',
    'Defender': 'DEF', 
    'Midfielder': 'MID',
    'Forward': 'FWD'
}

# Position names mapping (for backward compatibility)
POSITION_NAMES = {
    'GKP': 'Goalkeeper',
    'Goalkeeper': 'Goalkeeper',
    'DEF': 'Defender',
    'Defender': 'Defender',
    'MID': 'Midfielder',
    'Midfielder': 'Midfielder',
    'FWD': 'Forward',
    'Forward': 'Forward'
}

# Target variable
TARGET_COLUMN = 'actual_points'

# Feature categories for analysis - ENHANCED for position-specific features
FEATURE_CATEGORIES = {
    'Availability & Status': ['status_numeric', 'playing_probability', 'playing_chance_next', 
                             'playing_chance_this', 'availability_risk'],
    'Form & Performance': ['form_score', 'form_adjusted', 'consistency_score', 'form_trend',
                          'ppg_adjusted', 'overall_consistency'],
    'Rolling Performance': ['rolling_points_3', 'rolling_points_5', 'rolling_points_10',
                           'rolling_minutes_3', 'rolling_minutes_5', 'rolling_xg_3',
                           'rolling_xa_3', 'rolling_xgi_3'],
    'xG Metrics': ['xg_overperformance', 'xg_efficiency', 'xa_overperformance',
                  'xa_efficiency', 'xgi_overperformance', 'xgi_efficiency',
                  'expected_goals', 'expected_assists', 'expected_goal_involvements'],
    'Team & Fixture': ['team_strength_composite', 'strength', 'elo', 'position_fdr',
                      'base_fdr', 'is_home_next'],
    'ICT Index': ['influence_current', 'creativity_current', 'threat_current', 
                 'ict_index_current', 'bps_current'],
    'Position-Specific - Goalkeeper': ['saves_per_90', 'conceded_per_90', 'clean_sheet_prob',
                                      'save_percentage', 'penalty_save_rate', 'gk_bonus_per_90',
                                      'rolling_saves_3', 'rolling_conceded_3'],
    'Position-Specific - Defender': ['def_actions_per_90', 'tackles_per_90', 'def_clean_sheet_prob',
                                    'clearances_blocks_interceptions'],
    'Position-Specific - Midfielder': ['xa_per_90', 'goal_involvement_per_90', 'creativity_threat_ratio',
                                      'ct_index'],
    'Position-Specific - Forward': ['xg_per_90', 'goal_efficiency', 'threat_per_minute',
                                   'goal_scoring_probability'],
    'Basic Stats': ['minutes', 'total_points', 'goals_scored', 'assists', 
                   'clean_sheets', 'goals_conceded', 'bonus', 'saves',
                   'played_90', 'played_60', 'played_30', 'got_bonus']
}

# Features to EXCLUDE for RAW point prediction - STRICT VERSION
EXCLUDED_FEATURES_RAW = [
    # Cost features - ALL variants
    'now_cost', 'now_cost_current', 'cost_change_event', 'cost_change_event_current',
    'cost_change_trend', 'cost_increasing',
    
    # Value metrics - ALL variants
    'points_per_million', 'form_per_cost', 'expected_points_per_cost',
    'value_form', 'value_season', 'value_for_money', 'ownership_value',
    'value_form_current', 'value_season_current',
    
    # Ownership features - ALL variants
    'selected_by_percent', 'ownership_pct', 'high_ownership',
    'selected_by_percent_current',
    
    # Transfer features - ALL variants
    'transfers_in', 'transfers_out', 'transfers_in_event', 'transfers_out_event',
    'transfer_balance', 'transfer_net', 'transfer_momentum', 'popularity_pressure',
    
    # Data leakage risk - expected points for current gameweek
    'ep_this',
    
    # Status columns with _current suffix (except form_current)
    'influence_current', 'creativity_current', 'threat_current', 
    'ict_index_current', 'bps_current', 'defensive_contribution_current',
    
    # Problematic merged features
    '*_status',  # Will filter all columns ending with _status
]

# Performance thresholds for RAW point prediction
MAE_THRESHOLD = 1.5  # Target MAE for raw point prediction
R2_THRESHOLD = 0.3   # Minimum R² score

# Dynamic training configuration
TRAINING_WINDOW = 5  # Train on last 5 gameweeks
PREDICTION_LOOKAHEAD = 1  # Predict next gameweek
MIN_TRAINING_GWS = 3  # Minimum gameweeks needed for training

# Model file prefixes
MODEL_PREFIX_RAW = 'raw_'  # For raw point models
MODEL_PREFIX_VALUE = 'value_'  # For value-based models (if needed later)

# Position-specific minimum samples
MIN_SAMPLES_PER_POSITION = {
    'Goalkeeper': 20,
    'Defender': 50,
    'Midfielder': 50,
    'Forward': 30
}

# Optimization settings (for Phase 4)
OPTIMIZATION_BUDGET = 100.0  # £100M budget
SQUAD_SIZE = 15  # 15 players total
POSITION_LIMITS = {
    'Goalkeeper': (2, 2),    # min, max
    'Defender': (5, 5),
    'Midfielder': (5, 5),
    'Forward': (3, 3)
}
TEAM_LIMIT = 3  # Max 3 players from any single team