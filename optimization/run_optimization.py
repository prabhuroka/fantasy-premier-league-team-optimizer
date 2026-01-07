"""
Main script for running team optimization with chips and transfers 
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime
import glob
import re
import subprocess

from optimization.config import (
    BUDGET, POSITION_LIMITS, PREDICTIONS_FILE, OPTIMAL_PREDICTIONS_FILE,
    OPTIMIZATION_DIR, DATABASE_PATH, PREDICTIONS_DIR,
    FREE_TRANSFERS_PER_WEEK, POINTS_PER_TRANSFER
)
from ml_model.data_loader import DataLoader
from optimization.constraint_handler import FPLConstraintHandler, Player
from optimization.team_optimizer import TeamOptimizer
from optimization.transfer_planner import TransferPlanner, Transfer
from optimization.captain_selector import CaptainSelector, CaptainPick
from optimization.chip_strategist import ChipStrategist, ChipRecommendation


def load_current_team(filepath: str = None) -> Dict:
    """
    Load current team from file or create sample team
    
    Args:
        filepath: Path to team JSON file
        
    Returns:
        Dictionary with team data
    """
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            team_data = json.load(f)
            
            # Validate team structure
            required_fields = ['players', 'bank_balance', 'free_transfers', 'chips_available']
            for field in required_fields:
                if field not in team_data:
                    print(f"⚠ Warning: Missing field '{field}' in team file")
                    if field == 'players':
                        team_data[field] = []
                    elif field == 'bank_balance':
                        team_data[field] = 0.0
                    elif field == 'free_transfers':
                        team_data[field] = 1
                    elif field == 'chips_available':
                        team_data[field] = []
            
            return team_data
    
    # Return empty team structure
    return {
        'players': [],  # List of player IDs
        'bank_balance': 0.0,
        'free_transfers': 1,
        'chips_available': [],
        'total_points': 0,
        'overall_rank': 0
    }


def save_current_team(team_data: Dict, filepath: str = None) -> str:
    """
    Save current team to file
    
    Args:
        team_data: Team data dictionary
        filepath: Path to save file
        
    Returns:
        Path to saved file
    """
    if filepath is None:
        filepath = os.path.join(OPTIMIZATION_DIR, 'current_team.json')
    
    with open(filepath, 'w') as f:
        json.dump(team_data, f, indent=2)
    
    print(f"Team saved to: {filepath}")
    return filepath


def get_player_details(player_ids: List[int], predictions_df: pd.DataFrame) -> List[Dict]:
    """
    Get player details from predictions DataFrame
    
    Args:
        player_ids: List of player IDs
        predictions_df: Predictions DataFrame
        
    Returns:
        List of player details
    """
    player_details = []
    
    for player_id in player_ids:
        player_data = predictions_df[predictions_df['player_id'] == player_id]
        
        if not player_data.empty:
            player = player_data.iloc[0]
            player_details.append({
                'player_id': int(player_id),
                'name': player.get('web_name', f'Player {player_id}'),
                'position': player.get('position', 'Unknown'),
                'team_id': int(player.get('team_id', 0)),
                'team_name': player.get('team_name', f'Team {player.get("team_id", 0)}'),
                'cost': float(player.get('now_cost', 0)),
                'raw_predicted_points': float(player.get('raw_predicted_points', 0)),
                'predicted_points': float(player.get('predicted_points', 0)),
                'points_per_million': float(player.get('points_per_million', 0)),
                'selected_by_percent': float(player.get('selected_by_percent', 0)),
                'playing_probability': float(player.get('playing_probability', 1.0))
            })
        else:
            print(f"⚠ Player {player_id} not found in predictions")
    
    return player_details


def analyze_current_team(current_team_data: Dict, predictions_df: pd.DataFrame) -> Dict:
    """
    Analyze current team performance
    
    Args:
        current_team_data: Current team data
        predictions_df: Predictions DataFrame
        
    Returns:
        Current team analysis
    """
    if not current_team_data['players']:
        return {
            'expected_points': 0,
            'total_cost': 0,
            'bank_balance': current_team_data['bank_balance'],
            'is_valid': False,
            'errors': ['No players in current team']
        }
    
    # Get player details
    player_details = get_player_details(current_team_data['players'], predictions_df)
    
    if len(player_details) != len(current_team_data['players']):
        print(f"⚠ Only found {len(player_details)}/{len(current_team_data['players'])} current team players in predictions")
    
    # Convert to Player objects for constraint checking
    players = []
    for player_dict in player_details:
        try:
            player = Player(
                player_id=player_dict['player_id'],
                name=player_dict['name'],
                position=player_dict['position'],
                team_id=player_dict['team_id'],
                team_name=player_dict['team_name'],
                cost=player_dict['cost'],
                raw_predicted_points=player_dict['raw_predicted_points'],
                predicted_points=player_dict['predicted_points'],
                points_per_million=player_dict['points_per_million'],
                selected_by_percent=player_dict['selected_by_percent'],
                playing_probability=player_dict['playing_probability']
            )
            players.append(player)
        except Exception as e:
            print(f"⚠ Could not create Player object for {player_dict['player_id']}: {e}")
    
    # Validate team
    constraint_handler = FPLConstraintHandler()
    is_valid, errors = constraint_handler.validate_team(players)
    
    # Calculate expected points (simple - top 11 by predicted points)
    if players:
        sorted_players = sorted(players, key=lambda x: x.predicted_points, reverse=True)
        starting_xi = sorted_players[:11]
        expected_points = sum(p.predicted_points for p in starting_xi)
        total_cost = sum(p.cost for p in players)
    else:
        expected_points = 0
        total_cost = 0
    
    return {
        'expected_points': expected_points,
        'total_cost': total_cost,
        'bank_balance': current_team_data['bank_balance'],
        'free_transfers': current_team_data['free_transfers'],
        'chips_available': current_team_data['chips_available'],
        'player_count': len(players),
        'is_valid': is_valid,
        'errors': errors
    }


def check_and_run_ml_pipeline(gameweek: int = None) -> str:
    """
    Check if predictions exist, run ML pipeline if not
    
    Args:
        gameweek: Target gameweek (None for latest)
        
    Returns:
        Path to predictions file
    """
    predictions_dir = PREDICTIONS_DIR
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Determine which gameweek file to look for
    if gameweek:
        # Specific gameweek
        expected_file = os.path.join(predictions_dir, f'raw_points_predictions_gw{gameweek}.csv')
        print(f"  Looking for predictions for specific GW{gameweek}")
    else:
        # Latest gameweek - find the highest gameweek number
        pattern = os.path.join(predictions_dir, 'raw_points_predictions_gw*.csv')
        existing_files = glob.glob(pattern)
        
        if existing_files:
            # Extract gameweek numbers from filenames
            gws = []
            for file in existing_files:
                match = re.search(r'gw(\d+)', file.lower())
                if match:
                    gws.append(int(match.group(1)))
            
            if gws:
                latest_gw = max(gws)
                expected_file = os.path.join(predictions_dir, f'raw_points_predictions_gw{latest_gw}.csv')
                print(f"  Found existing predictions up to GW{latest_gw}")
            else:
                expected_file = os.path.join(predictions_dir, 'raw_points_predictions_gw1.csv')
                print(f"  No predictions found, will generate for GW1")
        else:
            expected_file = os.path.join(predictions_dir, 'raw_points_predictions_gw1.csv')
            print(f"  No predictions found, will generate for GW1")
    
    # Check if predictions already exist
    if os.path.exists(expected_file):
        print(f"  ✓ Found predictions: {os.path.basename(expected_file)}")
        return expected_file
    
    # Predictions don't exist, run ML pipeline
    print(f"  ⚠ Predictions not found: {os.path.basename(expected_file)}")
    print(f"  🔧 Running ML pipeline...")
    
    try:
        # Import and run ML pipeline
        ml_pipeline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        'ml_model', 'run_raw_point_pipeline.py')
        
        if not os.path.exists(ml_pipeline_path):
            print(f"  ✗ ML pipeline not found at: {ml_pipeline_path}")
            return ""
        
        # Run the ML pipeline with appropriate arguments
        if gameweek:
            # Specific gameweek - pass gameweek argument
            cmd = [sys.executable, ml_pipeline_path, "--predict-only", "--gameweek", str(gameweek)]
        else:
            # Latest gameweek - just predict-only (will auto-detect latest)
            cmd = [sys.executable, ml_pipeline_path, "--predict-only"]
        
        print(f"  Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(ml_pipeline_path))
        
        if result.returncode == 0:
            print(f"  ✓ ML pipeline completed successfully")
            
            # Check if expected file was created
            if os.path.exists(expected_file):
                return expected_file
            else:
                # Try to find any new predictions file
                pattern = os.path.join(predictions_dir, 'raw_points_predictions_gw*.csv')
                existing_files = glob.glob(pattern)
                if existing_files:
                    # Get the most recent file
                    existing_files.sort(key=os.path.getmtime, reverse=True)
                    print(f"  ✓ Found new predictions: {os.path.basename(existing_files[0])}")
                    return existing_files[0]
                else:
                    print(f"  ✗ ML pipeline didn't create predictions file")
                    return ""
        else:
            print(f"  ✗ ML pipeline failed with exit code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return ""
            
    except Exception as e:
        print(f"  ✗ Failed to run ML pipeline: {e}")
        return ""


def optimize_with_transfers(current_team_data: Dict, 
                          optimized_team: Dict,
                          predictions_df: pd.DataFrame) -> Dict:
    """
    Plan optimal transfers from current team to optimized team
    
    Args:
        current_team_data: Current team data
        optimized_team: Optimized team result
        predictions_df: Predictions DataFrame
        
    Returns:
        Transfer plan
    """
    # Get current team players
    current_player_details = get_player_details(current_team_data['players'], predictions_df)
    current_players = []
    for player_dict in current_player_details:
        try:
            player = Player(
                player_id=player_dict['player_id'],
                name=player_dict['name'],
                position=player_dict['position'],
                team_id=player_dict['team_id'],
                team_name=player_dict['team_name'],
                cost=player_dict['cost'],
                raw_predicted_points=player_dict['raw_predicted_points'],
                predicted_points=player_dict['predicted_points'],
                points_per_million=player_dict['points_per_million']
            )
            current_players.append(player)
        except Exception as e:
            print(f"⚠ Could not create Player object for current player {player_dict['player_id']}: {e}")
    
    # Get optimized team players - FIX: Check for both 'selected_players' and 'players'
    optimized_players = []
    
    # Check which key exists
    if 'selected_players' in optimized_team:
        players_list = optimized_team['selected_players']
    elif 'players' in optimized_team:
        players_list = optimized_team['players']
    else:
        print(f"⚠ No players found in optimized team")
        return {
            'num_transfers': 0,
            'point_hit': 0,
            'expected_gain': 0,
            'net_gain': 0,
            'transfers': []
        }
    
    for player_dict in players_list:
        try:
            if isinstance(player_dict, dict):
                player = Player(
                    player_id=player_dict['player_id'],
                    name=player_dict['name'],
                    position=player_dict['position'],
                    team_id=player_dict['team_id'],
                    team_name=player_dict['team_name'],
                    cost=player_dict['cost'],
                    raw_predicted_points=player_dict['raw_predicted_points'],
                    predicted_points=player_dict['predicted_points'],
                    points_per_million=player_dict['points_per_million']
                )
                optimized_players.append(player)
        except Exception as e:
            print(f"⚠ Could not create Player object for optimized player: {e}")
    
    # If no current players, return empty transfer plan
    if not current_players:
        return {
            'num_transfers': 0,
            'point_hit': 0,
            'expected_gain': 0,
            'net_gain': 0,
            'transfers': []
        }
    
    # Plan transfers
    try:
        transfer_planner = TransferPlanner(FPLConstraintHandler())
        
        transfers = transfer_planner.find_optimal_transfers(
            current_team=current_players,
            available_players=optimized_players,
            bank_balance=current_team_data['bank_balance'],
            free_transfers=current_team_data['free_transfers'],
            max_transfers=len(current_team_data['players'])  # Can replace entire team
        )
        
        # Calculate transfer impact
        transfer_impact = {
            'num_transfers': 0,
            'point_hit': 0,
            'expected_gain': 0,
            'net_gain': 0,
            'transfers': []
        }
        
        if transfers:
            # Take only transfers with positive net gain
            positive_transfers = [t for t in transfers if t.net_gain > 0]
            
            # Limit to free transfers + 1 (to avoid excessive point hits)
            max_allowed = current_team_data['free_transfers'] + 1
            selected_transfers = positive_transfers[:max_allowed]
            
            transfer_impact['num_transfers'] = len(selected_transfers)
            transfer_impact['point_hit'] = sum(t.point_hit for t in selected_transfers)
            transfer_impact['expected_gain'] = sum(t.expected_gain for t in selected_transfers)
            transfer_impact['net_gain'] = sum(t.net_gain for t in selected_transfers)
            
            # Format transfer details
            for transfer in selected_transfers:
                transfer_impact['transfers'].append({
                    'player_in': {
                        'id': transfer.player_in.player_id,
                        'name': transfer.player_in.name,
                        'position': transfer.player_in.position,
                        'cost': transfer.player_in.cost,
                        'expected_points': transfer.player_in.predicted_points
                    },
                    'player_out': {
                        'id': transfer.player_out.player_id,
                        'name': transfer.player_out.name,
                        'position': transfer.player_out.position,
                        'cost': transfer.player_out.cost,
                        'expected_points': transfer.player_out.predicted_points
                    },
                    'point_hit': transfer.point_hit,
                    'expected_gain': transfer.expected_gain,
                    'net_gain': transfer.net_gain,
                    'cost_difference': transfer.cost_difference
                })
        
        return transfer_impact
        
    except Exception as e:
        print(f"⚠ Transfer planning failed: {e}")
        return {
            'num_transfers': 0,
            'point_hit': 0,
            'expected_gain': 0,
            'net_gain': 0,
            'transfers': []
        }


def analyze_chip_strategy(current_team_data: Dict,
                         optimized_team: Dict,
                         gameweek: int) -> Dict:
    """
    Analyze chip strategy options - ENHANCED VERSION
    
    Args:
        current_team_data: Current team data
        optimized_team: Optimized team
        gameweek: Current gameweek
        
    Returns:
        Chip strategy analysis
    """
    chip_analysis = {
        'available_chips': current_team_data.get('chips_available', []),
        'recommendations': [],
        'best_chip': None,
        'chip_plan': {}  # Multi-GW chip plan
    }
    
    if not chip_analysis['available_chips']:
        return chip_analysis
    
    # Create Player objects for chip analysis
    optimized_players = []
    
    # FIX: Check for both 'selected_players' and 'players' keys
    if 'selected_players' in optimized_team:
        players_list = optimized_team['selected_players']
    elif 'players' in optimized_team:
        players_list = optimized_team['players']
    else:
        print(f"⚠ No players found in optimized team for chip analysis")
        return chip_analysis
    
    for player_dict in players_list:
        if isinstance(player_dict, dict):
            try:
                player = Player(
                    player_id=player_dict['player_id'],
                    name=player_dict['name'],
                    position=player_dict['position'],
                    team_id=player_dict['team_id'],
                    team_name=player_dict['team_name'],
                    cost=player_dict['cost'],
                    raw_predicted_points=player_dict['raw_predicted_points'],
                    predicted_points=player_dict['predicted_points'],
                    points_per_million=player_dict['points_per_million'],
                    playing_probability=player_dict.get('playing_probability', 1.0)
                )
                optimized_players.append(player)
            except Exception as e:
                print(f"⚠ Could not create Player object for chip analysis: {e}")
    
    # Use enhanced chip strategist
    chip_strategist = ChipStrategist()
    
    # Get single best chip recommendation for current week
    single_recommendation = chip_strategist.recommend_single_chip(
        current_team=optimized_players,
        available_chips=chip_analysis['available_chips'],
        gameweek=gameweek
    )
    
    if single_recommendation:
        chip_analysis['best_chip'] = {
            'chip_name': single_recommendation.chip_name,
            'recommended_gw': single_recommendation.recommended_gw,
            'expected_gain': single_recommendation.expected_gain,
            'confidence': single_recommendation.confidence,
            'reason': single_recommendation.reason,
            'alternative_chips': single_recommendation.alternative_chips,
            'risks': single_recommendation.risks
        }
        
        # Also get multi-gameweek chip plan
        chip_plan = chip_strategist.compare_chip_strategies(
            current_team=optimized_players,
            available_chips=chip_analysis['available_chips'],
            gameweek=gameweek,
            horizon=5  # Plan next 5 gameweeks
        )
        
        chip_analysis['chip_plan'] = {
            str(gw): {
                'chip': rec.chip_name if rec else None,
                'expected_gain': rec.expected_gain if rec else 0.0,
                'reason': rec.reason if rec else None
            }
            for gw, rec in chip_plan.items()
        }
    
    return chip_analysis


def compare_teams(current_analysis: Dict,
                  optimized_analysis: Dict,
                  transfer_plan: Dict,
                  chip_analysis: Dict) -> Dict:
    """
    Compare current team vs optimized team
    
    Args:
        current_analysis: Current team analysis
        optimized_analysis: Optimized team analysis
        transfer_plan: Transfer plan
        chip_analysis: Chip strategy analysis
        
    Returns:
        Comparison results
    """
    comparison = {
        'current_team': {
            'expected_points': current_analysis['expected_points'],
            'total_cost': current_analysis['total_cost'],
            'bank_balance': current_analysis['bank_balance'],
            'is_valid': current_analysis['is_valid']
        },
        'optimized_team': {
            'expected_points': optimized_analysis['expected_points'],
            'total_cost': optimized_analysis['total_cost'],
            'bank_balance': optimized_analysis['bank_balance'],
            'is_valid': optimized_analysis['is_valid']
        },
        'improvement': {
            'points_gain': optimized_analysis['expected_points'] - current_analysis['expected_points'],
            'points_gain_with_transfers': 0,
            'points_gain_with_chips': 0
        },
        'recommendation': 'keep_current_team',  # Default
        'reason': '',
        'actions_needed': []
    }
    
    # Calculate improvement with transfers
    if transfer_plan['net_gain'] > 0:
        comparison['improvement']['points_gain_with_transfers'] = (
            comparison['improvement']['points_gain'] + transfer_plan['net_gain']
        )
    
    # Calculate improvement with chips
    if chip_analysis['best_chip']:
        comparison['improvement']['points_gain_with_chips'] = (
            comparison['improvement']['points_gain_with_transfers'] + 
            chip_analysis['best_chip']['expected_gain']
        )
    
    # Determine recommendation
    points_gain = comparison['improvement']['points_gain']
    
    if points_gain > 5:
        comparison['recommendation'] = 'use_wildcard'
        comparison['reason'] = 'Significant improvement available with new team'
        comparison['actions_needed'].append('Use Wildcard chip')
    
    elif points_gain > 2 and transfer_plan['net_gain'] > 0:
        comparison['recommendation'] = 'make_transfers'
        comparison['reason'] = f'Net gain of {transfer_plan["net_gain"]:.1f} points available with transfers'
        comparison['actions_needed'].append(f'Make {transfer_plan["num_transfers"]} transfers')
    
    elif chip_analysis['best_chip'] and chip_analysis['best_chip']['expected_gain'] > 3:
        comparison['recommendation'] = 'use_chip'
        comparison['reason'] = f'Chip usage expected to gain {chip_analysis["best_chip"]["expected_gain"]:.1f} points'
        comparison['actions_needed'].append(f'Use {chip_analysis["best_chip"]["chip_name"]} chip')
    
    elif points_gain > 0:
        comparison['recommendation'] = 'consider_optimization'
        comparison['reason'] = f'Small improvement of {points_gain:.1f} points available'
    
    else:
        comparison['recommendation'] = 'keep_current_team'
        comparison['reason'] = 'Current team is already optimal or better'
    
    return comparison


def optimize_complete_team(gameweek: int = None,
                          current_team_file: str = None,
                          predictions_file: str = None,
                          method: str = 'mip',
                          save_result: bool = True) -> Dict:
    """
    Run complete team optimization with transfers and chips
    
    Args:
        gameweek: Gameweek to optimize for
        current_team_file: Path to current team JSON
        predictions_file: Path to RAW predictions CSV
        method: Optimization method ('mip' or 'heuristic')
        save_result: Whether to save result to file
        
    Returns:
        Complete optimization result
    """
    print("\n" + "="*80)
    print("FPL TEAM OPTIMIZATION - COMPLETE ANALYSIS")
    print("="*80)
    
    # Determine gameweek if not provided
    if gameweek is None:
        # Try to get from predictions file name
        if predictions_file:
            match = re.search(r'gw(\d+)', predictions_file.lower())
            if match:
                gameweek = int(match.group(1))
    
    if gameweek is None:
        data_loader = DataLoader()
        latest_gw = data_loader.get_latest_gameweek()
        gameweek = latest_gw + 1
    
    print(f"\n🎯 OPTIMIZING FOR GAMEWEEK {gameweek}")
    
    # Load or generate predictions file
    if predictions_file is None:
        predictions_file = check_and_run_ml_pipeline(gameweek)
        
        if not predictions_file:
            return {"error": f"Could not find or generate predictions for GW{gameweek}"}
    
    if not os.path.exists(predictions_file):
        return {"error": f"Predictions file not found: {predictions_file}"}
    
    print(f"  Predictions: {os.path.basename(predictions_file)}")
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    print(f"  Loaded predictions for {len(predictions_df)} players")
    
    # Load current team
    current_team_data = load_current_team(current_team_file)
    print(f"\n📊 CURRENT TEAM ANALYSIS")
    print(f"  Players: {len(current_team_data['players'])}")
    print(f"  Bank balance: £{current_team_data['bank_balance']:.1f}M")
    print(f"  Free transfers: {current_team_data['free_transfers']}")
    print(f"  Available chips: {', '.join(current_team_data['chips_available']) if current_team_data['chips_available'] else 'None'}")
    
    # Analyze current team
    current_analysis = analyze_current_team(current_team_data, predictions_df)
    print(f"  Expected points: {current_analysis['expected_points']:.1f}")
    print(f"  Total cost: £{current_analysis['total_cost']:.1f}M")
    print(f"  Team valid: {'✅' if current_analysis['is_valid'] else '❌'}")
    if not current_analysis['is_valid']:
        print(f"  Errors: {', '.join(current_analysis['errors'][:3])}")
    
    # Step 1: Optimize team from scratch
    print(f"\n1️⃣ OPTIMIZING NEW TEAM")
    
    constraint_handler = FPLConstraintHandler()
    optimizer = TeamOptimizer(constraint_handler)
    
    # Load predictions for optimizer
    optimizer.load_raw_predictions(predictions_file)
    
    optimization_result = optimizer.optimize_team_with_transfers(
        predictions_file=predictions_file,
        current_team=current_team_data['players'] if current_team_data['players'] else None,
        bank_balance=current_team_data['bank_balance'],
        free_transfers=current_team_data['free_transfers'],
        method=method
    )
    
    if 'error' in optimization_result:
        print(f"❌ Optimization failed: {optimization_result['error']}")
        return optimization_result
    
    print(f"  ✅ Optimized team: {optimization_result['expected_points']:.1f} expected points")
    print(f"  ✅ Total cost: £{optimization_result['total_cost']:.1f}M")
    print(f"  ✅ Transfers needed: {optimization_result['num_transfers']}")
    
    # Get selected players - FIX: Check for both 'selected_players' and 'players'
    if 'selected_players' in optimization_result:
        selected_players_dict = optimization_result['selected_players']
    elif 'players' in optimization_result:
        selected_players_dict = optimization_result['players']
    else:
        print(f"❌ No players found in optimization result")
        return {"error": "No players selected in optimization"}
    
    if not selected_players_dict:
        print(f"❌ No players selected in optimization")
        return {"error": "No players selected"}
    
    # Get starting XI and bench - FIX: Check for both keys
    if 'starting_xi_ids' in optimization_result:
        starting_xi = optimization_result['starting_xi_ids']
    elif 'starting_xi' in optimization_result:
        starting_xi = optimization_result['starting_xi']
    else:
        starting_xi = []
    
    # Get bench players
    bench = []
    for player in selected_players_dict:
        if isinstance(player, dict):
            player_id = player.get('player_id')
            if player_id and player_id not in starting_xi:
                bench.append(player_id)
    
    # Step 2: Plan transfers
    print(f"\n2️⃣ PLANNING TRANSFERS")
    
    transfer_plan = optimize_with_transfers(
        current_team_data=current_team_data,
        optimized_team=optimization_result,
        predictions_df=predictions_df
    )
    
    print(f"  ✅ Transfers planned: {transfer_plan['num_transfers']}")
    if transfer_plan['num_transfers'] > 0:
        print(f"  ✅ Point hit: {transfer_plan['point_hit']}")
        print(f"  ✅ Net gain: {transfer_plan['net_gain']:.1f} points")
    
    # Step 3: Analyze chip strategy
    print(f"\n3️⃣ ANALYZING CHIP STRATEGY")
    
    chip_analysis = analyze_chip_strategy(
        current_team_data=current_team_data,
        optimized_team=optimization_result,
        gameweek=gameweek
    )
    
    print(f"  ✅ Available chips: {len(chip_analysis['available_chips'])}")
    if chip_analysis['best_chip']:
        print(f"  ✅ Best chip: {chip_analysis['best_chip']['chip_name']}")
        print(f"  ✅ Expected gain: {chip_analysis['best_chip']['expected_gain']:.1f} points")
    
    # Step 4: Select captain
    print(f"\n4️⃣ SELECTING CAPTAIN")
    
    try:
        # Convert selected players to Player objects
        selected_players_objects = []
        for player_dict in selected_players_dict:
            try:
                if isinstance(player_dict, dict):
                    player = Player(
                        player_id=int(player_dict.get('player_id', 0)),
                        name=str(player_dict.get('name', 'Unknown')),
                        position=str(player_dict.get('position', 'Unknown')),
                        team_id=int(player_dict.get('team_id', 0)),
                        team_name=str(player_dict.get('team_name', f"Team {player_dict.get('team_id', 0)}")),
                        cost=float(player_dict.get('cost', 5.0)),
                        raw_predicted_points=float(player_dict.get('raw_predicted_points', 0)),
                        predicted_points=float(player_dict.get('predicted_points', 0)),
                        points_per_million=float(player_dict.get('points_per_million', 0))
                    )
                    selected_players_objects.append(player)
            except Exception as e:
                print(f"   ⚠ Could not convert player: {e}")
        
        # Get starting XI players for captain selection
        starting_xi_players = [p for p in selected_players_objects if p.player_id in starting_xi]
        
        if starting_xi_players:
            captain_selector = CaptainSelector()
            captain_pick = captain_selector.select_captain(starting_xi_players)
            
            captain_info = {
                'player_id': captain_pick.player.player_id,
                'player_name': captain_pick.player.name,
                'predicted_points': captain_pick.predicted_points,
                'captain_points': captain_pick.captain_points,
                'confidence': captain_pick.selection_confidence,
                'alternatives': [p.player_id for p in captain_pick.alternative_players]
            }
            
            # Select vice-captain
            vice_pick = captain_selector.select_vice_captain(starting_xi_players, captain_pick.player.player_id)
            vice_captain_info = {
                'player_id': vice_pick.player.player_id,
                'player_name': vice_pick.player.name,
                'predicted_points': vice_pick.predicted_points
            }
            
            print(f"  ✅ Captain: {captain_pick.player.name} ({captain_pick.captain_points:.1f} expected points)")
            print(f"  ✅ Vice-captain: {vice_pick.player.name}")
        else:
            print(f"  ⚠ No starting XI players found")
            captain_info = {}
            vice_captain_info = {}
            
    except Exception as e:
        print(f"  ⚠ Captain selection failed: {e}")
        captain_info = {}
        vice_captain_info = {}
    
    # Step 5: Compare teams
    print(f"\n5️⃣ COMPARING TEAMS")
    
    optimized_analysis = {
        'expected_points': optimization_result['expected_points'],
        'total_cost': optimization_result['total_cost'],
        'bank_balance': optimization_result.get('bank_balance', 0),
        'is_valid': optimization_result.get('validation', {}).get('is_valid', True)
    }
    
    comparison = compare_teams(
        current_analysis=current_analysis,
        optimized_analysis=optimized_analysis,
        transfer_plan=transfer_plan,
        chip_analysis=chip_analysis
    )
    
    print(f"  ✅ Current team: {comparison['current_team']['expected_points']:.1f} points")
    print(f"  ✅ Optimized team: {comparison['optimized_team']['expected_points']:.1f} points")
    print(f"  ✅ Improvement: {comparison['improvement']['points_gain']:.1f} points")
    print(f"  ✅ Recommendation: {comparison['recommendation'].replace('_', ' ').title()}")
    print(f"  ✅ Reason: {comparison['reason']}")
    
    # Step 6: Bench strength calculation
    print(f"\n6️⃣ BENCH STRENGTH")
    
    try:
        bench_players = [p for p in selected_players_objects if p.player_id in bench]
        bench_strength = sum(p.predicted_points for p in bench_players)
        bench_availability = np.mean([p.playing_probability for p in bench_players]) if bench_players else 0
        
        print(f"  ✅ Bench strength: {bench_strength:.1f} expected points")
        print(f"  ✅ Bench availability: {bench_availability:.1%}")
        
    except Exception as e:
        print(f"  ⚠ Bench calculation failed: {e}")
        bench_strength = 0.0
        bench_availability = 0.0
    
    # Step 7: Compile final result
    print(f"\n7️⃣ COMPILING RESULTS")
    
    final_result = {
        'gameweek': gameweek,
        'timestamp': datetime.now().isoformat(),
        'optimization_method': method,
        'separated_approach': True,
        'raw_points_source': os.path.basename(predictions_file),
        
        'current_team_analysis': current_analysis,
        'optimized_team': optimization_result,
        'transfer_plan': transfer_plan,
        'chip_analysis': chip_analysis,
        'captain': captain_info,
        'vice_captain': vice_captain_info,
        'comparison': comparison,
        
        'lineup': {
            'starting_xi': starting_xi,
            'bench': bench,
            'bench_strength': bench_strength,
            'bench_availability': bench_availability
        },
        
        'recommendation': {
            'action': comparison['recommendation'],
            'reason': comparison['reason'],
            'actions_needed': comparison['actions_needed'],
            'expected_improvement': comparison['improvement']['points_gain_with_transfers']
        },
        
        'players': selected_players_dict
    }
    
    # Generate optimal predictions file for reference
    try:
        optimal_file = optimizer.generate_optimal_predictions_file(
            final_result, predictions_file
        )
        if optimal_file:
            final_result['optimal_predictions_file'] = optimal_file
    except Exception as e:
        print(f"  ⚠ Could not generate optimal predictions file: {e}")
    
    # Save result
    if save_result:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_optimization_gw{gameweek}.json"
        filepath = os.path.join(OPTIMIZATION_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(final_result, f, indent=2, default=str)
        
        print(f"  ✅ Results saved to: {filepath}")
    
    # Print summary
    print_summary(final_result)
    
    return final_result


def print_summary(result: Dict):
    """
    Print optimization summary
    
    Args:
        result: Optimization result
    """
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    if 'error' in result:
        print(f"\n❌ ERROR: {result['error']}")
        return
    
    # Current team
    current = result['current_team_analysis']
    print(f"\n📊 CURRENT TEAM:")
    print(f"  Expected points: {current['expected_points']:.1f}")
    print(f"  Total cost: £{current['total_cost']:.1f}M")
    print(f"  Bank balance: £{current['bank_balance']:.1f}M")
    print(f"  Valid: {'✅' if current['is_valid'] else '❌'}")
    
    # Optimized team
    optimized = result['optimized_team']
    print(f"\n🎯 OPTIMIZED TEAM:")
    print(f"  Expected points: {optimized['expected_points']:.1f}")
    print(f"  Total cost: £{optimized['total_cost']:.1f}M")
    print(f"  Bank balance: £{optimized.get('bank_balance', 0):.1f}M")
    
    if 'formation' in optimized:
        formation = optimized['formation']
        print(f"  Formation: {formation.get('formation', 'N/A')}")
    
    print(f"  Transfers needed: {optimized['num_transfers']}")
    
    # Transfers
    transfers = result['transfer_plan']
    if transfers['num_transfers'] > 0:
        print(f"\n🔄 RECOMMENDED TRANSFERS ({transfers['num_transfers']}):")
        print(f"  Point hit: {transfers['point_hit']}")
        print(f"  Expected gain: {transfers['expected_gain']:.1f} points")
        print(f"  Net gain: {transfers['net_gain']:.1f} points")
        
        for i, transfer in enumerate(transfers['transfers'][:3], 1):
            print(f"  {i}. {transfer['player_out']['name']} → {transfer['player_in']['name']}")
            print(f"     Gain: {transfer['net_gain']:.1f} points")
    
    # Chips
    chips = result['chip_analysis']
    if chips['best_chip']:
        print(f"\n🎲 CHIP RECOMMENDATION:")
        print(f"  Use {chips['best_chip']['chip_name']} in GW{chips['best_chip']['recommended_gw']}")
        print(f"  Expected gain: {chips['best_chip']['expected_gain']:.1f} points")
        print(f"  Reason: {chips['best_chip']['reason']}")
    
    # Captain
    captain = result.get('captain', {})
    if captain:
        print(f"\n👑 CAPTAIN SELECTION")
        print(f"  Captain: {captain.get('player_name', 'Unknown')}")
        print(f"  RAW Points: {captain.get('predicted_points', 0):.1f}")
        print(f"  Captain Points: {captain.get('captain_points', 0):.1f}")
        print(f"  Confidence: {captain.get('confidence', 0):.0%}")
    
    # Recommendation
    rec = result['recommendation']
    print(f"\n💡 RECOMMENDATION: {rec['action'].replace('_', ' ').title()}")
    print(f"  Reason: {rec['reason']}")
    
    if rec['actions_needed']:
        print(f"  Actions needed:")
        for action in rec['actions_needed']:
            print(f"    • {action}")
    
    # Comparison
    comp = result['comparison']
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print(f"  Points gain: {comp['improvement']['points_gain']:.1f}")
    if transfers['net_gain'] > 0:
        print(f"  With transfers: {comp['improvement']['points_gain_with_transfers']:.1f}")
    if chips['best_chip']:
        print(f"  With chips: {comp['improvement']['points_gain_with_chips']:.1f}")


def main():
    """Main entry point for complete optimization"""
    parser = argparse.ArgumentParser(description='FPL Complete Team Optimization')
    parser.add_argument('--gameweek', '-gw', type=int, 
                       help='Gameweek to optimize for (None for latest + 1)')
    parser.add_argument('--team-file', '-t', type=str, 
                       default='user_team.json',
                       help='Path to current team JSON file')
    parser.add_argument('--predictions-file', '-p', type=str,
                       help='Path to RAW predictions CSV file')
    parser.add_argument('--method', '-m', choices=['mip', 'heuristic'], 
                       default='mip',
                       help='Optimization method')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    parser.add_argument('--update-team', action='store_true',
                       help='Update current team file with optimized team')
    parser.add_argument('--quick', action='store_true',
                       help='Quick optimization (heuristic method)')
    parser.add_argument('--example-team', action='store_true', 
                       help='Use example team from example_current_team.json')
    
    args = parser.parse_args()
    
    # Set method based on quick flag
    method = 'heuristic' if args.quick else args.method
    
    # Use example team if requested
    if args.example_team:
        example_file = os.path.join(OPTIMIZATION_DIR, 'example_current_team.json')
        if not os.path.exists(example_file):
            # Create example team
            example_team = {
                "players": [32, 443, 225, 109, 112, 107, 382, 125, 47, 417,
                          64, 430, 287, 411, 350],
                "bank_balance": 2.5,
                "free_transfers": 1,
                "chips_available": ["wildcard", "free_hit", "triple_captain", "bench_boost"],
                "total_points": 987,
                "overall_rank": 125432
            }
            with open(example_file, 'w') as f:
                json.dump(example_team, f, indent=2)
            print(f"Created example team at: {example_file}")
        
        args.team_file = example_file
    
    # Run optimization
    result = optimize_complete_team(
        gameweek=args.gameweek,
        current_team_file=args.team_file,
        predictions_file=args.predictions_file,
        method=method,
        save_result=not args.no_save
    )
    
    # Update team file if requested
    if args.update_team and 'optimized_team' in result:
        # Create updated team data
        optimized = result['optimized_team']
        current_data = load_current_team(args.team_file)
        
        updated_team = {
            'players': optimized.get('selected_player_ids', []),
            'bank_balance': optimized.get('bank_balance', 0),
            'free_transfers': 1,  # Reset to 1 for next week
            'chips_available': current_data.get('chips_available', []),
            'total_points': current_data.get('total_points', 0) + optimized.get('expected_points', 0),
            'overall_rank': current_data.get('overall_rank', 0)
        }
        
        # Remove used chip if recommended
        chip_rec = result.get('chip_analysis', {}).get('best_chip', {})
        if chip_rec and chip_rec['chip_name'] in updated_team['chips_available']:
            updated_team['chips_available'].remove(chip_rec['chip_name'])
            print(f"\nRemoved {chip_rec['chip_name']} from available chips")
        
        save_current_team(updated_team, args.team_file)
    
    # Exit code based on success
    if 'error' in result:
        return 1
    return 0


if __name__ == "__main__":
    success = main()
    sys.exit(success)