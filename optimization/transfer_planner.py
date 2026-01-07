"""
Transfer planning and optimization
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from optimization.config import FREE_TRANSFERS_PER_WEEK, POINTS_PER_TRANSFER
from optimization.constraint_handler import Player, FPLConstraintHandler


@dataclass
class Transfer:
    """Transfer data class"""
    player_in: Player
    player_out: Player
    point_hit: int
    expected_gain: float
    net_gain: float
    cost_difference: float


class TransferPlanner:
    """Plan optimal transfers considering point hits and expected gains"""
    
    def __init__(self, constraint_handler: FPLConstraintHandler):
        """
        Initialize TransferPlanner
        
        Args:
            constraint_handler: FPLConstraintHandler instance
        """
        self.constraint_handler = constraint_handler
        
    def find_optimal_transfers(self, current_team: List[Player],
                              available_players: List[Player],
                              bank_balance: float,
                              free_transfers: int = 1,
                              max_transfers: int = 15) -> List[Transfer]:
        """
        Find optimal transfers to improve team
        
        Args:
            current_team: Current team players
            available_players: Available players to bring in
            bank_balance: Current bank balance
            free_transfers: Number of free transfers
            max_transfers: Maximum number of transfers to consider
            
        Returns:
            List of optimal transfers
        """
        # Sort current team by predicted points (weakest first)
        current_team_sorted = sorted(current_team, key=lambda x: x.predicted_points)
        
        # Sort available players by predicted points per cost (best value first)
        available_sorted = sorted(
            available_players,
            key=lambda x: x.predicted_points / max(x.cost, 0.1),  # Avoid division by zero
            reverse=True
        )
        
        transfers = []
        used_budget = bank_balance
        transfers_made = 0
        
        for current_player in current_team_sorted:
            if transfers_made >= max_transfers:
                break
            
            # Find best replacement for this player
            best_transfer = None
            best_net_gain = -float('inf')
            
            for new_player in available_sorted:
                # Skip if same player
                if new_player.player_id == current_player.player_id:
                    continue
                
                # Skip if new player is already in team
                if any(p.player_id == new_player.player_id for p in current_team):
                    continue
                
                # Check if we can afford the transfer
                cost_diff = new_player.cost - current_player.cost
                if cost_diff > used_budget:
                    continue
                
                # Check if transfer maintains position balance
                # Count positions in team after transfer
                temp_team = [p for p in current_team if p.player_id != current_player.player_id]
                temp_team.append(new_player)
                
                is_valid, _ = self.constraint_handler.validate_team(temp_team)
                if not is_valid:
                    continue
                
                # Calculate expected gain
                expected_gain = new_player.predicted_points - current_player.predicted_points
                
                # Calculate point hit (beyond free transfers)
                transfers_needed = transfers_made + 1
                point_hit = max(0, (transfers_needed - free_transfers)) * POINTS_PER_TRANSFER
                
                # Calculate net gain
                net_gain = expected_gain - point_hit
                
                if net_gain > best_net_gain:
                    best_net_gain = net_gain
                    best_transfer = Transfer(
                        player_in=new_player,
                        player_out=current_player,
                        point_hit=point_hit,
                        expected_gain=expected_gain,
                        net_gain=net_gain,
                        cost_difference=cost_diff
                    )
            
            if best_transfer and best_transfer.net_gain > 0:
                transfers.append(best_transfer)
                used_budget -= best_transfer.cost_difference
                transfers_made += 1
                
                # Update current team for next iteration
                current_team = [p for p in current_team if p.player_id != best_transfer.player_out.player_id]
                current_team.append(best_transfer.player_in)
        
        # Sort transfers by net gain
        transfers.sort(key=lambda x: x.net_gain, reverse=True)
        
        return transfers
    
    def plan_transfers_with_horizon(self, current_team: List[Player],
                                   available_players: List[Player],
                                   bank_balance: float,
                                   free_transfers: int = 1,
                                   horizon: int = 3) -> Dict[int, List[Transfer]]:
        """
        Plan transfers for multiple gameweeks
        
        Args:
            current_team: Current team players
            available_players: Available players (with multi-GW predictions)
            bank_balance: Current bank balance
            free_transfers: Number of free transfers
            horizon: Number of gameweeks to plan for
            
        Returns:
            Dictionary mapping gameweek to list of transfers
        """
        # This is a simplified version - in practice would need multi-GW predictions
        transfers_by_gw = {}
        
        # Plan for each gameweek
        for gw in range(1, horizon + 1):
            # Filter players with predictions for this GW
            # (In reality, would filter by gw column in predictions)
            gw_players = available_players  # Simplified
            
            # Calculate transfers for this GW
            transfers = self.find_optimal_transfers(
                current_team=current_team,
                available_players=gw_players,
                bank_balance=bank_balance,
                free_transfers=free_transfers,
                max_transfers=free_transfers + 1  # Allow 1 extra for planning
            )
            
            transfers_by_gw[gw] = transfers
            
            # Update team for next GW (apply transfers)
            if transfers:
                # Apply the best transfer for this GW
                best_transfer = transfers[0]
                current_team = [p for p in current_team if p.player_id != best_transfer.player_out.player_id]
                current_team.append(best_transfer.player_in)
                bank_balance -= best_transfer.cost_difference
                free_transfers = max(1, free_transfers - 1)  # Use free transfer
        
        return transfers_by_gw
    
    def evaluate_transfer_strategy(self, transfers: List[Transfer],
                                 current_team: List[Player],
                                 new_team: List[Player]) -> Dict[str, float]:
        """
        Evaluate a transfer strategy
        
        Args:
            transfers: List of transfers to make
            current_team: Current team before transfers
            new_team: Team after transfers
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate metrics
        total_point_hit = sum(t.point_hit for t in transfers)
        total_expected_gain = sum(t.expected_gain for t in transfers)
        total_net_gain = sum(t.net_gain for t in transfers)
        
        # Calculate team improvement
        current_points = sum(p.predicted_points for p in current_team)
        new_points = sum(p.predicted_points for p in new_team)
        team_improvement = new_points - current_points
        
        # Calculate cost changes
        current_cost = sum(p.cost for p in current_team)
        new_cost = sum(p.cost for p in new_team)
        cost_change = new_cost - current_cost
        
        return {
            'num_transfers': len(transfers),
            'total_point_hit': total_point_hit,
            'total_expected_gain': total_expected_gain,
            'total_net_gain': total_net_gain,
            'team_improvement': team_improvement,
            'current_team_points': current_points,
            'new_team_points': new_points,
            'current_team_cost': current_cost,
            'new_team_cost': new_cost,
            'cost_change': cost_change,
            'points_per_transfer': total_net_gain / max(len(transfers), 1),
            'efficiency': total_net_gain / max(total_point_hit, 0.1)
        }