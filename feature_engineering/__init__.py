"""
Feature Engineering Module for FPL Simulator

This module creates ML-ready features from raw FPL data:
- Rolling performance averages
- Fixture difficulty ratings
- Injury risk assessments
- Value metrics
- Team context features
"""

from .feature_builder import FeatureBuilder
from .fixture_difficulty import FixtureDifficultyCalculator
from .injury_processor import InjuryProcessor
from .feature_pipeline import FeaturePipeline

__all__ = [
    'FeatureBuilder',
    'FixtureDifficultyCalculator',
    'InjuryProcessor',
    'FeaturePipeline'
]

__version__ = '1.0.0'
