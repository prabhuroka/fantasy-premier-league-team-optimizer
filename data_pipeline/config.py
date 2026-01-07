"""
Configuration settings for the FPL Simulator
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATABASE_PATH = os.path.join(PROCESSED_DATA_DIR, 'sqlite', 'fpl_simulator.db')

# Seasons
CURRENT_SEASON = "2025-2026"

# GitHub repository
GITHUB_REPO_URL = "https://github.com/olbauday/FPL-Elo-Insights"
GITHUB_BRANCH = "main"
GITHUB_SEASON_PATH = f"data/{CURRENT_SEASON}"

# Database settings
SQLITE_PRAGMAS = {
    'journal_mode': 'WAL',
    'cache_size': -1 * 10000,  # 10MB
    'foreign_keys': 1,
    'synchronous': 'NORMAL'
}

# Data processing
MIN_MINUTES_THRESHOLD = 30  # Minimum minutes to consider player active