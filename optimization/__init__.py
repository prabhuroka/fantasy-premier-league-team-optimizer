"""
Team Optimization Module for FPL Simulator - PHASE 4

This module optimizes FPL team selection using the separated approach:
- Layer 1 (Phase 3): RAW point predictions (performance only)
- Layer 2 (Phase 4): Value optimization combining RAW points with cost

Key components:
- TeamOptimizer: Main optimization engine using Mixed Integer Programming
- ConstraintHandler: Enforces FPL rules (positions, budget, team limits)
- TransferPlanner: Recommends optimal transfers
- CaptainSelector: Identifies best captain and vice-captain
- ChipStrategist: Recommends chip usage strategy
"""

from .config import (
    BUDGET, POSITION_LIMITS, TEAM_LIMIT, SQUAD_SIZE,
    STARTING_XI_SIZE, OPTIMIZATION_METHOD,
    MIN_PLAYING_PROBABILITY, MIN_AVAILABILITY_FOR_STARTERS
)
from .constraint_handler import FPLConstraintHandler, Player
from .team_optimizer import TeamOptimizer
from .transfer_planner import TransferPlanner, Transfer
from .captain_selector import CaptainSelector, CaptainPick
from .chip_strategist import ChipStrategist, ChipRecommendation
from .run_optimization import optimize_complete_team, load_current_team

__all__ = [
    'FPLConstraintHandler',
    'Player',
    'TeamOptimizer',
    'TransferPlanner',
    'Transfer',
    'CaptainSelector',
    'CaptainPick',
    'ChipStrategist',
    'ChipRecommendation',
    'optimize_complete_team',
    'load_current_team',
    'BUDGET',
    'POSITION_LIMITS',
    'TEAM_LIMIT',
    'SQUAD_SIZE'
]

__version__ = '4.0.0'
