"""
Main optimization engine for FPL team selection
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pulp
import itertools
from datetime import datetime
import json
import os

from optimization.config import (
    BUDGET, POSITION_LIMITS, TEAM_LIMIT, SQUAD_SIZE,
    STARTING_XI_SIZE, OPTIMIZATION_METHOD, MAX_SOLUTION_TIME,
    SOLUTION_TOLERANCE, OPTIMIZATION_DIR,
    MIN_PLAYING_PROBABILITY, MIN_AVAILABILITY_FOR_STARTERS
)
from optimization.constraint_handler import FPLConstraintHandler, Player


class TeamOptimizer:
    """Optimize FPL team selection using separated approach (RAW points + cost)"""
    
    def __init__(self, constraint_handler: FPLConstraintHandler = None):
        """
        Initialize TeamOptimizer for separated approach
        
        Args:
            constraint_handler: FPLConstraintHandler instance
        """
        self.constraint_handler = constraint_handler or FPLConstraintHandler()
        self.players = []
        self.player_dict = {}
        
    def load_raw_predictions(self, predictions_file: str,
                            min_playing_probability: float = MIN_PLAYING_PROBABILITY,
                            min_features_used: int = 10) -> List[Player]:
        """
        Load RAW predictions from Phase 3 and convert to Player objects
        
        Args:
            predictions_file: Path to RAW predictions CSV
            min_playing_probability: Minimum playing probability to consider
            min_features_used: Minimum features used in prediction
            
        Returns:
            List of Player objects for optimization
        """
        try:
            df = pd.read_csv(predictions_file)
            players = []
            
            # Standardize position names
            position_mapping = {
                'GKP': 'Goalkeeper',
                'Goalkeeper': 'Goalkeeper',
                'DEF': 'Defender',
                'Defender': 'Defender',
                'MID': 'Midfielder',
                'Midfielder': 'Midfielder',
                'FWD': 'Forward',
                'Forward': 'Forward'
            }
            
            for _, row in df.iterrows():
                # Skip players with low availability
                playing_prob = row.get('playing_probability', row.get('availability', 1.0))
                if playing_prob < min_playing_probability:
                    continue
                
                # Skip if insufficient features used
                features_used = row.get('features_used', 20)
                if features_used < min_features_used:
                    continue
                
                # Get position (standardized)
                position = str(row.get('position', 'Unknown'))
                position = position_mapping.get(position, position)
                
                # Get cost (convert from FPL units to £M)
                now_cost = row.get('now_cost', 50)  # FPL units (5.0 = £5.0M)
                cost = now_cost / 10.0  # Convert to £M
                
                # Get RAW predicted points
                raw_points = row.get('raw_predicted_points', row.get('predicted_points', 0))
                
                # Adjust points based on position (separated approach: RAW points × position weight)
                position_weight = {
                    'Goalkeeper': 1.0,
                    'Defender': 1.0,
                    'Midfielder': 1.1,
                    'Forward': 1.2
                }.get(position, 1.0)
                
                adjusted_points = raw_points * position_weight
                
                # Get value metric (points per million)
                points_per_million = row.get('points_per_million', 0)
                if points_per_million == 0 and cost > 0:
                    points_per_million = raw_points / cost
                
                player = Player(
                    player_id=int(row.get('player_id', 0)),
                    name=row.get('web_name', f"Player {row.get('player_id', 'Unknown')}"),
                    position=position,
                    team_id=int(row.get('team_id', 0)),
                    team_name=row.get('team_name', f"Team {row.get('team_id', 'Unknown')}"),
                    cost=float(now_cost),
                    raw_predicted_points=float(raw_points),
                    predicted_points=float(adjusted_points),
                    points_per_million=float(points_per_million),
                    selected_by_percent=float(row.get('selected_by_percent', 0)),
                    playing_probability=float(playing_prob),
                    injury_risk=float(row.get('injury_risk', 0)),
                    rotation_risk=float(row.get('rotation_risk', 0)),
                    fixture_difficulty=float(row.get('fixture_difficulty', 3.0)),
                    is_injured=bool(row.get('is_injured', False)),
                    is_suspended=bool(row.get('is_suspended', False)),
                    features_used=int(features_used),
                    availability_risk=float(row.get('availability_risk', 0))
                )
                
                players.append(player)
            
            self.players = players
            self.player_dict = {p.player_id: p for p in players}
            
            print(f"Loaded {len(players)} players from {predictions_file}")
            print(f"  Average cost: £{np.mean([p.cost for p in players]):.1f}M")
            print(f"  Average predicted points: {np.mean([p.predicted_points for p in players]):.1f}")
            print(f"  Average playing probability: {np.mean([p.playing_probability for p in players]):.1%}")
            
            return players
            
        except Exception as e:
            print(f"Error loading predictions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def solve_mip_with_starting_xi(self, 
                                  current_team: List[int] = None,
                                  bank_balance: float = 0.0,
                                  free_transfers: int = 1,
                                  value_weight: float = 0.1, 
                                  use_value_metric: bool = True) -> Dict[str, Any]:
        """
        Solve team optimization using Mixed Integer Programming
        WITH SEPARATED APPROACH (RAW points × position weight) + value optimization
        
        Args:
            current_team: List of player IDs in current team
            bank_balance: Current bank balance
            free_transfers: Number of free transfers available
            use_value_metric: Whether to use points_per_million in objective
            
        Returns:
            Dictionary with optimization results
        """
        if not self.players:
            return {"error": "No players loaded"}
        
        # Create optimization problem
        prob = pulp.LpProblem("FPL_Team_Optimization_Separated", pulp.LpMaximize)
        
        # Decision variables:
        # x_i = 1 if player i is selected in squad, 0 otherwise
        # y_i = 1 if player i is in starting XI, 0 otherwise
        x = pulp.LpVariable.dicts("squad", 
                                 [p.player_id for p in self.players], 
                                 0, 1, pulp.LpBinary)
        
        y = pulp.LpVariable.dicts("starting", 
                                 [p.player_id for p in self.players], 
                                 0, 1, pulp.LpBinary)
        
        # Additional variables for transfers
        transfer_in = {}
        transfer_out = {}
        
        if current_team:
            current_player_ids = set(current_team)
            available_player_ids = set(p.player_id for p in self.players)
            
            # Create transfer variables
            for player_id in available_player_ids:
                if player_id not in current_player_ids:
                    transfer_in[player_id] = pulp.LpVariable(f"transfer_in_{player_id}", 0, 1, pulp.LpBinary)
            
            for player_id in current_player_ids:
                if player_id in available_player_ids:
                    transfer_out[player_id] = pulp.LpVariable(f"transfer_out_{player_id}", 0, 1, pulp.LpBinary)
        
        # OBJECTIVE: Maximize total value (points_per_million) for squad
        # PLUS maximize predicted points for starting XI
        # This implements the separated approach: RAW performance + value optimization
        
        if use_value_metric:
            predicted_points_weight = 1.0  # Primary weight for points
            value_weight_adjusted = value_weight  # Secondary weight for value

            # Use points_per_million for value optimization
            prob += (
            pulp.lpSum([
                p.predicted_points * y[p.player_id] * predicted_points_weight
                for p in self.players
            ]) + 
            pulp.lpSum([
                p.points_per_million * x[p.player_id] * value_weight_adjusted
                for p in self.players
            ])
        )
        else:
            # Use predicted points only (fallback)
            prob += pulp.lpSum([
                p.predicted_points * y[p.player_id]  # Maximize starting XI points
                for p in self.players
            ])
        
        # CONSTRAINTS
        
        # 1. Squad size constraint (15 players)
        prob += pulp.lpSum([x[p.player_id] for p in self.players]) == SQUAD_SIZE
        
        # 2. Starting XI size constraint (11 players)
        prob += pulp.lpSum([y[p.player_id] for p in self.players]) == STARTING_XI_SIZE
        
        # 3. Starting XI must be subset of squad
        for p in self.players:
            prob += y[p.player_id] <= x[p.player_id]
        
        # 4. Budget constraint
        available_budget = BUDGET + bank_balance
        prob += pulp.lpSum([p.cost * x[p.player_id] for p in self.players]) <= available_budget
        
        # 5. Squad position constraints (2-5-5-3)
        for position, (min_limit, max_limit) in POSITION_LIMITS.items():
            position_players = [p for p in self.players if p.position == position]
            prob += pulp.lpSum([x[p.player_id] for p in position_players]) >= min_limit
            prob += pulp.lpSum([x[p.player_id] for p in position_players]) <= max_limit
        
        # 6. Starting XI position constraints
        # At least 1 goalkeeper in starting XI
        gk_players = [p for p in self.players if p.position == 'Goalkeeper']
        if gk_players:
            prob += pulp.lpSum([y[p.player_id] for p in gk_players]) >= 1
        
        # At least 3 defenders in starting XI
        def_players = [p for p in self.players if p.position == 'Defender']
        if len(def_players) >= 3:
            prob += pulp.lpSum([y[p.player_id] for p in def_players]) >= 3
        
        # At least 2 midfielders in starting XI
        mid_players = [p for p in self.players if p.position == 'Midfielder']
        if len(mid_players) >= 2:
            prob += pulp.lpSum([y[p.player_id] for p in mid_players]) >= 2
        
        # At least 1 forward in starting XI
        fwd_players = [p for p in self.players if p.position == 'Forward']
        if fwd_players:
            prob += pulp.lpSum([y[p.player_id] for p in fwd_players]) >= 1
        
        # 7. Team limit constraint (max 3 players from any team)
        team_ids = list(set(p.team_id for p in self.players))
        for team_id in team_ids:
            team_players = [p for p in self.players if p.team_id == team_id]
            prob += pulp.lpSum([x[p.player_id] for p in team_players]) <= TEAM_LIMIT
        
        # 8. Availability constraint for starting XI
        # Starting players should have good playing probability
        for p in self.players:
            if p.playing_probability < MIN_AVAILABILITY_FOR_STARTERS:
                prob += y[p.player_id] <= 0  # Cannot be in starting XI
                # But can be on bench
                prob += x[p.player_id] <= 1  # Can be in squad
        
        # 9. Transfer constraints (if current team provided)
        if current_team:
            # Number of transfers constraint
            num_transfers = pulp.lpSum([transfer_in[pid] for pid in transfer_in.keys()])
            prob += num_transfers == pulp.lpSum([transfer_out[pid] for pid in transfer_out.keys()])
            
            # Free transfers constraint
            max_transfers = free_transfers + len(current_team)  # Can make unlimited with hits
            prob += num_transfers <= max_transfers
            
            # Transfer relationship constraints
            for player_id in set(transfer_in.keys()) | set(transfer_out.keys()):
                if player_id in transfer_in and player_id in transfer_out:
                    # Can't transfer in and out the same player
                    prob += transfer_in[player_id] + transfer_out[player_id] <= 1
                
                if player_id in transfer_in:
                    # If transferring in, player must be selected in squad
                    prob += transfer_in[player_id] <= x[player_id]
                
                if player_id in transfer_out:
                    # If transferring out, player must not be selected in squad
                    prob += transfer_out[player_id] <= 1 - x[player_id]
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=MAX_SOLUTION_TIME, gapRel=SOLUTION_TOLERANCE)
        prob.solve(solver)
        
        if pulp.LpStatus[prob.status] != "Optimal":
            print(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
            return {"error": f"Optimization failed: {pulp.LpStatus[prob.status]}"}
        
        # Extract solution
        selected_player_ids = []
        starting_xi_ids = []
        
        for p in self.players:
            if pulp.value(x[p.player_id]) > 0.5:
                selected_player_ids.append(p.player_id)
            
            if pulp.value(y[p.player_id]) > 0.5:
                starting_xi_ids.append(p.player_id)
        
        # Calculate transfers
        transfers_in = []
        transfers_out = []
        
        if current_team:
            for player_id, var in transfer_in.items():
                if pulp.value(var) > 0.5:
                    transfers_in.append(player_id)
            
            for player_id, var in transfer_out.items():
                if pulp.value(var) > 0.5:
                    transfers_out.append(player_id)
        
        # Get selected players
        selected_players = [p for p in self.players if p.player_id in selected_player_ids]
        starting_xi_players = [p for p in self.players if p.player_id in starting_xi_ids]
        
        # Calculate metrics
        total_cost = sum(p.cost for p in selected_players)
        expected_points = sum(p.predicted_points for p in starting_xi_players)
        squad_points = sum(p.predicted_points for p in selected_players)
        
        # Calculate value metrics
        total_value_score = sum(p.points_per_million for p in selected_players)
        avg_playing_prob = np.mean([p.playing_probability for p in starting_xi_players])
        
        # Calculate position distribution
        position_distribution = {}
        for player in selected_players:
            position = player.position
            position_distribution[position] = position_distribution.get(position, 0) + 1
        
        # Calculate team distribution
        team_distribution = {}
        for player in selected_players:
            team_id = player.team_id
            team_distribution[team_id] = team_distribution.get(team_id, 0) + 1
        
        # Calculate formation
        starting_positions = [p.position for p in starting_xi_players]
        formation = {
            'goalkeepers': starting_positions.count('Goalkeeper'),
            'defenders': starting_positions.count('Defender'),
            'midfielders': starting_positions.count('Midfielder'),
            'forwards': starting_positions.count('Forward'),
            'formation': f"{starting_positions.count('Defender')}-{starting_positions.count('Midfielder')}-{starting_positions.count('Forward')}"
        }
        
        return {
            "selected_player_ids": selected_player_ids,
            "selected_players": selected_players,
            "starting_xi_ids": starting_xi_ids,
            "starting_xi_players": starting_xi_players,
            "total_cost": total_cost,
            "bank_balance": available_budget - total_cost,
            "expected_points": expected_points,
            "squad_points": squad_points,
            "total_value_score": total_value_score,
            "avg_playing_probability": avg_playing_prob,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "num_transfers": len(transfers_in),
            "position_distribution": position_distribution,
            "team_distribution": team_distribution,
            "formation": formation,
            "optimization_status": pulp.LpStatus[prob.status]
        }
    
    def optimize_team_with_transfers(self, predictions_file: str,
                                    current_team: List[int] = None,
                                    bank_balance: float = 0.0,
                                    free_transfers: int = 1,
                                    method: str = None) -> Dict[str, Any]:
        """
        Main optimization method for separated approach
        
        Args:
            predictions_file: Path to RAW predictions CSV
            current_team: List of player IDs in current team
            bank_balance: Current bank balance
            free_transfers: Number of free transfers available
            method: Optimization method ('mip' or 'heuristic')
            
        Returns:
            Optimization results
        """
        # Load RAW predictions
        self.load_raw_predictions(predictions_file)
        
        # Use specified method or default
        method = method or OPTIMIZATION_METHOD
        
        print(f"\nOptimizing team with {len(self.players)} available players...")
        print(f"  Budget: £{BUDGET + bank_balance:.1f}M")
        print(f"  Free transfers: {free_transfers}")
        print(f"  Method: {method.upper()}")
        
        if method == 'mip':
            result = self.solve_mip_with_starting_xi(current_team, bank_balance, free_transfers)
        else:
            # Fall back to heuristic if needed
            result = self.solve_heuristic(current_team, bank_balance, free_transfers)
        
        # Add timestamp and additional info
        result['timestamp'] = datetime.now().isoformat()
        result['method'] = method
        result['gameweek'] = self._extract_gameweek_from_filename(predictions_file)
        
        # Convert Player objects to dictionaries for easier handling
        if 'selected_players' in result:
            players_dict = []
            for player in result['selected_players']:
                if isinstance(player, Player):
                    player_dict = {
                        'player_id': player.player_id,
                        'name': player.name,
                        'position': player.position,
                        'team_id': player.team_id,
                        'team_name': player.team_name,
                        'cost': player.cost,
                        'raw_predicted_points': player.raw_predicted_points,
                        'predicted_points': player.predicted_points,
                        'points_per_million': player.points_per_million,
                        'selected_by_percent': player.selected_by_percent,
                        'playing_probability': player.playing_probability,
                        'injury_risk': player.injury_risk,
                        'rotation_risk': player.rotation_risk,
                        'fixture_difficulty': player.fixture_difficulty,
                        'is_injured': player.is_injured,
                        'is_suspended': player.is_suspended,
                        'features_used': player.features_used,
                        'availability_risk': player.availability_risk
                    }
                    players_dict.append(player_dict)
                else:
                    players_dict.append(player)
            
            result['selected_players'] = players_dict
        
        # Validate the optimized team
        if 'selected_players' in result and 'starting_xi_ids' in result:
            selected_players = [Player(**p) if isinstance(p, dict) else p for p in result['selected_players']]
            is_valid, errors = self.constraint_handler.validate_team(
                selected_players, 
                result['starting_xi_ids'],
                enforce_availability=True
            )
            
            result['validation'] = {
                'is_valid': is_valid,
                'errors': errors
            }
            
            if not is_valid:
                print(f"⚠ Team validation failed with {len(errors)} errors")
                for error in errors[:3]:
                    print(f"  - {error}")
            else:
                print(f"✅ Team validation passed")
        
        return result
    
    def solve_heuristic(self, current_team: List[int] = None,
                       bank_balance: float = 0.0,
                       free_transfers: int = 1) -> Dict[str, Any]:
        """
        Simple heuristic fallback solution
        """
        print("Using heuristic fallback solution...")
        
        # Sort by value (points_per_million) first
        sorted_players = sorted(self.players, key=lambda x: x.points_per_million, reverse=True)
        
        selected_players = []
        total_cost = 0
        position_counts = {'Goalkeeper': 0, 'Defender': 0, 'Midfielder': 0, 'Forward': 0}
        team_counts = {}
        
        for player in sorted_players:
            # Check position limits
            if position_counts[player.position] >= POSITION_LIMITS[player.position][1]:
                continue
            
            # Check team limits
            team_counts[player.team_id] = team_counts.get(player.team_id, 0)
            if team_counts[player.team_id] >= TEAM_LIMIT:
                continue
            
            # Check budget
            if total_cost + player.cost > BUDGET + bank_balance:
                continue
            
            # Add player
            selected_players.append(player)
            total_cost += player.cost
            position_counts[player.position] += 1
            team_counts[player.team_id] += 1
            
            # Stop when we have full squad
            if len(selected_players) == SQUAD_SIZE:
                break
        
        # Select starting XI (top 11 by predicted points)
        starting_xi_players = sorted(selected_players, key=lambda x: x.predicted_points, reverse=True)[:STARTING_XI_SIZE]
        starting_xi_ids = [p.player_id for p in starting_xi_players]
        
        expected_points = sum(p.predicted_points for p in starting_xi_players)
        squad_points = sum(p.predicted_points for p in selected_players)
        
        return {
            "selected_player_ids": [p.player_id for p in selected_players],
            "selected_players": selected_players,
            "starting_xi_ids": starting_xi_ids,
            "starting_xi_players": starting_xi_players,
            "total_cost": total_cost,
            "bank_balance": BUDGET + bank_balance - total_cost,
            "expected_points": expected_points,
            "squad_points": squad_points,
            "transfers_in": [],
            "transfers_out": [],
            "num_transfers": 0,
            "optimization_status": "Heuristic solution"
        }
    
    def _extract_gameweek_from_filename(self, filename: str) -> int:
        """Extract gameweek number from filename"""
        import re
        match = re.search(r'gw(\d+)', filename.lower())
        if match:
            return int(match.group(1))
        return 1
    
    def save_optimization_result(self, result: Dict[str, Any], 
                               filename: str = None) -> str:
        """
        Save optimization result to file
        
        Args:
            result: Optimization result dictionary
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gw = result.get('gameweek', 'unknown')
            filename = f"optimization_gw{gw}_{timestamp}.json"
        
        filepath = os.path.join(OPTIMIZATION_DIR, filename)
        
        # Convert Player objects to dictionaries for JSON serialization
        if 'selected_players' in result:
            result['selected_players'] = [
                {
                    'player_id': p.get('player_id', p.player_id if hasattr(p, 'player_id') else 0),
                    'name': p.get('name', getattr(p, 'name', 'Unknown')),
                    'position': p.get('position', getattr(p, 'position', 'Unknown')),
                    'team_id': p.get('team_id', getattr(p, 'team_id', 0)),
                    'team_name': p.get('team_name', getattr(p, 'team_name', 'Unknown')),
                    'cost': p.get('cost', getattr(p, 'cost', 0)),
                    'raw_predicted_points': p.get('raw_predicted_points', getattr(p, 'raw_predicted_points', 0)),
                    'predicted_points': p.get('predicted_points', getattr(p, 'predicted_points', 0)),
                    'points_per_million': p.get('points_per_million', getattr(p, 'points_per_million', 0))
                }
                for p in result['selected_players']
            ]
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Optimization result saved to: {filepath}")
        return filepath
    
    def generate_optimal_predictions_file(self, result: Dict[str, Any],
                                        predictions_file: str = None) -> str:
        """
        Generate a CSV file with optimal predictions for Phase 4
        
        Args:
            result: Optimization result dictionary
            predictions_file: Original predictions file path
            
        Returns:
            Path to generated optimal predictions file
        """
        if 'selected_players' not in result:
            print("No selected players in result")
            return ""
        
        # Load original predictions to get all columns
        if predictions_file and os.path.exists(predictions_file):
            original_df = pd.read_csv(predictions_file)
            data = []
            for player in result['selected_players']:
                if isinstance(player, dict):
                    data.append({
                        'player_id': player['player_id'],
                        'web_name': player['name'],
                        'position': player['position'],
                        'team_id': player['team_id'],
                        'team_name': player['team_name'],
                        'now_cost': player['cost'] * 10,  # Convert back to FPL units
                        'predicted_points': player['predicted_points'],
                        'raw_predicted_points': player['raw_predicted_points'],
                        'points_per_million': player['points_per_million'],
                        'is_optimized': 1,
                        'in_starting_xi': 1 if player['player_id'] in result.get('starting_xi_ids', []) else 0
                    })
            
            optimal_df = pd.DataFrame(data)
        else:
            # Mark optimized players in original dataframe
            optimized_ids = [p['player_id'] for p in result['selected_players']]
            starting_xi_ids = set(result.get('starting_xi_ids', []))
            
            original_df['is_optimized'] = original_df['player_id'].isin(optimized_ids).astype(int)
            original_df['in_starting_xi'] = original_df['player_id'].isin(starting_xi_ids).astype(int)
            optimal_df = original_df
        
        # Save optimal predictions
        gameweek = result.get('gameweek', 19)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OPTIMIZATION_DIR, f'optimal_predictions_gw{gameweek}_{timestamp}.csv')
        
        optimal_df.to_csv(output_file, index=False)
        print(f"Optimal predictions saved to: {output_file}")
        
        return output_file