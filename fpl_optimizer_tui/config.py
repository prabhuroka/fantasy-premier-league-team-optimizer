"""
Configuration for FPL Optimizer TUI
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Base paths
BASE_DIR = project_root
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
OPTIMIZATION_DIR = DATA_DIR / "optimization"
SAVED_TEAM_PATH = OPTIMIZATION_DIR / "example_current_team.json"

# Ensure directories exist
OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)

# FPL Rules
FPL_RULES = {
    'budget': 100.0,
    'squad_size': 15,
    'starting_xi': 11,
    'bench_size': 4,
    'position_limits': {
        'Goalkeeper': (2, 2),    # min, max
        'Defender': (5, 5),
        'Midfielder': (5, 5),
        'Forward': (3, 3)
    },
    'team_limit': 3,  # Max players from one team
    'formation_min': {
        'Goalkeeper': 1,
        'Defender': 3,
        'Midfielder': 2,
        'Forward': 1
    }
}

# Colors for positions
POSITION_COLORS = {
    'Goalkeeper': 'bright_blue',
    'Defender': 'green',
    'Midfielder': 'yellow',
    'Forward': 'red'
}

POSITION_EMOJI = {
    'Goalkeeper': '🧤',
    'Defender': '🛡️',
    'Midfielder': '⚽',
    'Forward': '🎯'
}

# Optimization command
OPTIMIZATION_CMD = [
    sys.executable,
    str(BASE_DIR / "optimization" / "run_optimization.py"),
    "--example-team"
]