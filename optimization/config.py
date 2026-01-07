"""
Configuration for FPL team optimization
"""
import os
from datetime import datetime

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
OPTIMIZATION_DIR = os.path.join(DATA_DIR, 'optimization')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# Database path - IMPORT FROM DATA PIPELINE CONFIG
try:
    from data_pipeline.config import DATABASE_PATH
except ImportError:
    # Fallback if data pipeline not available
    DATABASE_PATH = os.path.join(PROCESSED_DATA_DIR, 'sqlite', 'fpl_simulator.db')

# FPL Rules and Constraints - UPDATED FOR SEPARATED APPROACH
BUDGET = 100.0  # £100 million
POSITION_LIMITS = {
    'Goalkeeper': (2, 2),    # min, max
    'Defender': (5, 5),
    'Midfielder': (5, 5),
    'Forward': (3, 3)
}

TEAM_LIMIT = 3  # Max 3 players from any single team
SQUAD_SIZE = 15
STARTING_XI_SIZE = 11

# Optimization settings
OPTIMIZATION_METHOD = 'mip'  # 'mip' (Mixed Integer Programming) or 'heuristic'
MAX_SOLUTION_TIME = 30  # seconds
SOLUTION_TOLERANCE = 0.01  # 1% tolerance for near-optimal solutions

# Transfer settings
FREE_TRANSFERS_PER_WEEK = 1
POINTS_PER_TRANSFER = 4  # Points deducted for each transfer beyond free transfers
WILDCARD_WEEKS = [1, 20]  # Typical wildcard usage weeks
TRIPLE_CAPTAIN_WEEKS = []  # Weeks with double gameweeks

# Position-specific weights (for separated approach - combine RAW points with position factors)
POSITION_WEIGHTS = {
    'Goalkeeper': 1.0,
    'Defender': 1.0,
    'Midfielder': 1.1,  # Midfielders get bonus for goals/assists
    'Forward': 1.2      # Forwards get bonus for goals
}

# Captain multipliers
CAPTAIN_MULTIPLIER = 2.0
VICE_CAPTAIN_MULTIPLIER = 1.0  # No multiplier, just backup

# Risk factors
INJURY_RISK_PENALTY = 0.3
ROTATION_RISK_PENALTY = 0.2
FIXTURE_DIFFICULTY_WEIGHT = 0.15

# Availability thresholds (from comprehensive features)
MIN_PLAYING_PROBABILITY = 0.6  # Minimum playing probability for key players
MIN_AVAILABILITY_FOR_STARTERS = 0.7  # Minimum for starting XI
MAX_INJURY_RISK = 0.5  # Maximum injury risk to consider

# Performance targets
TARGET_POINTS_PER_GAMEWEEK = 60
MINIMUM_TEAM_VALUE = 95.0  # Minimum team value to maintain

# File paths
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'latest_raw_points_predictions_gw{gameweek}.csv')
OPTIMAL_PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'optimal_predictions_gw{gameweek}.csv')
SAVED_TEAMS_FILE = os.path.join(OPTIMIZATION_DIR, 'saved_teams.json')
OPTIMIZATION_HISTORY_FILE = os.path.join(OPTIMIZATION_DIR, 'optimization_history.csv')