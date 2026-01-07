"""
Enhanced configuration for feature engineering
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
os.makedirs(FEATURES_DIR, exist_ok=True)

# Import database path from data pipeline
try:
    from data_pipeline.config import DATABASE_PATH
except ImportError:
    # Fallback if data pipeline not available
    DATABASE_PATH = os.path.join(DATA_DIR, 'processed', 'sqlite', 'fpl_simulator.db')

# Feature engineering settings
ROLLING_WINDOWS = [3, 5, 10]  # Gameweeks for rolling averages
FORM_GAMEWEEKS = 5            # Gameweeks to consider for form
FIXTURE_LOOKAHEAD = 5         # Number of future fixtures to consider

# Position mapping
POSITION_MAPPING = {
    'GKP': 'Goalkeeper',
    'DEF': 'Defender',
    'MID': 'Midfielder',
    'FWD': 'Forward'
}

# New feature categories
HOME_AWAY_FEATURES = [
    'home_avg_points', 'away_avg_points', 'home_games_played',
    'away_games_played', 'home_advantage'
]

TEAM_STRENGTH_FEATURES = [
    'strength', 'strength_overall_home', 'strength_overall_away',
    'strength_attack_home', 'strength_attack_away',
    'strength_defence_home', 'strength_defence_away',
    'elo', 'elo_normalized', 'team_strength_composite',
    'attack_defense_balance_home', 'attack_defense_balance_away'
]

XG_BASED_FEATURES = [
    'xg_overperformance', 'xg_efficiency',
    'xa_overperformance', 'xa_efficiency',
    'xgi_overperformance', 'xgi_efficiency',
    'xg_per_minute', 'xg_consistency'
]

FIXTURE_DIFFICULTY_FEATURES = [
    'position_fdr', 'base_fdr', 'position_fdr_adjustment',
    'next_5_avg_fdr'
]

# Minimum thresholds
MIN_MINUTES_THRESHOLD = 30    # Minimum minutes to consider active
MIN_GAMEWEEKS_FOR_ROLLING = 3 # Minimum gameweeks needed for rolling features