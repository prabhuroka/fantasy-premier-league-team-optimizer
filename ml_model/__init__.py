"""
ML Model Module for FPL Simulator

This module implements machine learning models for FPL point prediction
using a separated approach:
- Layer 1: Predict RAW POINTS (performance only, no cost/value features)
- Layer 2: Value optimization (separate optimization phase)
"""

from .config import (
    MODELS_DIR, PREDICTIONS_DIR, DATABASE_PATH,
    RANDOM_SEED, TEST_SIZE, EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS, XGBOOST_PARAMS,
    MAX_FEATURES_PER_MODEL, MIN_FEATURE_IMPORTANCE,
    POSITIONS, POSITION_ABBREV, POSITION_NAMES,
    TARGET_COLUMN, FEATURE_CATEGORIES,
    MAE_THRESHOLD, R2_THRESHOLD,
    TRAINING_WINDOW, PREDICTION_LOOKAHEAD, MIN_TRAINING_GWS
)

from .data_loader import DataLoader
from .feature_selector import FeatureSelector
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_analyzer import ModelAnalyzer
from .raw_prediction_generator import RawPredictionGenerator

__all__ = [
    'DataLoader',
    'FeatureSelector',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelAnalyzer',
    'RawPredictionGenerator',
    'MODELS_DIR',
    'PREDICTIONS_DIR',
    'DATABASE_PATH',
    'POSITIONS',
    'TARGET_COLUMN'
]

__version__ = '2.0.0'  #Updated
