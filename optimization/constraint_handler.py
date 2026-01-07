"""
Handle FPL rules and constraints for team optimization
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import json

from optimization.config import (
    BUDGET, POSITION_LIMITS, TEAM_LIMIT, SQUAD_SIZE,
    STARTING_XI_SIZE, POSITION_WEIGHTS,
    MIN_PLAYING_PROBABILITY, MIN_AVAILABILITY_FOR_STARTERS,
    MAX_INJURY_RISK, CAPTAIN_MULTIPLIER
)


@dataclass
class Player:
    """Player data class for optimization - UPDATED FOR SEPARATED APPROACH"""
    player_id: int
    name: str
    position: str
    team_id: int
    team_name: str
    cost: float  # in £M (now_cost/10)
    raw_predicted_points: float  # RAW points from Phase 3
    predicted_points: float  # Adjusted points for optimization
    points_per_million: float  # Value metric
    selected_by_percent: float = 0.0
    playing_probability: float = 1.0  # From comprehensive features
    injury_risk: float = 0.0
    rotation_risk: float = 0.0
    fixture_difficulty: float = 3.0
    is_injured: bool = False
    is_suspended: bool = False
    features_used: int = 0  # From prediction generation
    availability_risk: float = 0.0  # From comprehensive features


class FPLConstraintHandler:
    """Handle all FPL rules and constraints - UPDATED FOR SEPARATED APPROACH"""
    
    def __init__(self):
        self.position_limits = POSITION_LIMITS
        self.budget = BUDGET
        self.team_limit = TEAM_LIMIT
        self.squad_size = SQUAD_SIZE
        self.starting_xi_size = STARTING_XI_SIZE
        
    def validate_team(self, players: List[Player], 
                     starting_xi: List[int] = None,
                     enforce_availability: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate if a team complies with FPL rules - UPDATED WITH AVAILABILITY
        
        Args:
            players: List of Player objects
            starting_xi: List of player IDs in starting XI
            enforce_availability: Whether to check availability constraints
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # 1. Check squad size
        if len(players) != self.squad_size:
            errors.append(f"Squad must have exactly {self.squad_size} players, has {len(players)}")
        
        # 2. Check budget
        total_cost = sum(p.cost for p in players)
        if total_cost > self.budget:
            errors.append(f"Total cost £{total_cost:.1f}M exceeds budget £{self.budget:.1f}M")
        
        # 3. Check position limits
        position_counts = {}
        for position in self.position_limits.keys():
            position_counts[position] = 0
        
        for player in players:
            if player.position in position_counts:
                position_counts[player.position] += 1
        
        for position, (min_limit, max_limit) in self.position_limits.items():
            count = position_counts[position]
            if count < min_limit or count > max_limit:
                errors.append(f"Need {min_limit}-{max_limit} {position}s, have {count}")
        
        # 4. Check team limits
        team_counts = {}
        for player in players:
            team_id = player.team_id
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        for team_id, count in team_counts.items():
            if count > self.team_limit:
                team_name = players[0].team_name if players else f"Team {team_id}"
                errors.append(f"Cannot have more than {self.team_limit} players from {team_name} (have {count})")
        
        # 5. Check starting XI if provided
        if starting_xi:
            if len(starting_xi) != self.starting_xi_size:
                errors.append(f"Starting XI must have {self.starting_xi_size} players, has {len(starting_xi)}")
            
            # Check formation validity (at least 1 GK, 3 DEF, 2 MID, 1 FWD)
            starting_positions = []
            for player_id in starting_xi:
                player = next((p for p in players if p.player_id == player_id), None)
                if player:
                    starting_positions.append(player.position)
            
            gk_count = starting_positions.count('Goalkeeper')
            def_count = starting_positions.count('Defender')
            mid_count = starting_positions.count('Midfielder')
            fwd_count = starting_positions.count('Forward')
            
            if gk_count < 1:
                errors.append("Starting XI must have at least 1 Goalkeeper")
            if def_count < 3:
                errors.append("Starting XI must have at least 3 Defenders")
            if mid_count < 2:
                errors.append("Starting XI must have at least 2 Midfielders")
            if fwd_count < 1:
                errors.append("Starting XI must have at least 1 Forward")
        
        # 6. Check for injured/suspended players in starting XI
        if starting_xi and enforce_availability:
            for player_id in starting_xi:
                player = next((p for p in players if p.player_id == player_id), None)
                if player:
                    if player.is_injured:
                        errors.append(f"{player.name} is injured but in starting XI")
                    if player.is_suspended:
                        errors.append(f"{player.name} is suspended but in starting XI")
                    if player.playing_probability < MIN_AVAILABILITY_FOR_STARTERS:
                        errors.append(f"{player.name} has low playing probability ({player.playing_probability:.0%}) but in starting XI")
        
        return len(errors) == 0, errors
    
    def calculate_expected_points(self, players: List[Player], 
                                 starting_xi: List[int],
                                 captain_id: int = None,
                                 vice_captain_id: int = None,
                                 consider_availability: bool = True) -> float:
        """
        Calculate expected points for a team - UPDATED FOR SEPARATED APPROACH
        
        Args:
            players: List of Player objects
            starting_xi: List of player IDs in starting XI
            captain_id: Captain player ID
            vice_captain_id: Vice-captain player ID
            consider_availability: Whether to adjust for playing probability
            
        Returns:
            Expected points total
        """
        total_points = 0.0
        
        for player in players:
            if player.player_id in starting_xi:
                # Use adjusted points (already includes position weights in optimization)
                points = player.predicted_points
                
                # Apply captain multiplier
                if player.player_id == captain_id:
                    points *= CAPTAIN_MULTIPLIER
                elif player.player_id == vice_captain_id:
                    # Vice-captain gets points if captain doesn't play
                    captain_player = next((p for p in players if p.player_id == captain_id), None)
                    if captain_player and consider_availability:
                        captain_play_prob = captain_player.playing_probability
                        points = points * (1.0 - captain_play_prob)
                
                # Adjust for availability if enabled
                if consider_availability:
                    points *= player.playing_probability
                
                # Adjust for fixture difficulty
                fixture_adjustment = (5.0 - player.fixture_difficulty) / 4.0  # 1 = easy, 5 = hard
                points *= (1.0 + 0.1 * (1 - fixture_adjustment))  # Bonus for easy fixtures
                
                total_points += points
        
        return total_points
    
    def get_formation(self, starting_xi: List[int], players: List[Player]) -> str:
        """
        Get formation string (e.g., "4-4-2")
        
        Args:
            starting_xi: List of player IDs in starting XI
            players: List of Player objects
            
        Returns:
            Formation string
        """
        # Count positions in starting XI
        position_map = {}
        for player_id in starting_xi:
            player = next((p for p in players if p.player_id == player_id), None)
            if player:
                position_map[player.position] = position_map.get(player.position, 0) + 1
        
        # Create formation string
        gk = position_map.get('Goalkeeper', 0)
        defenders = position_map.get('Defender', 0)
        midfielders = position_map.get('Midfielder', 0)
        forwards = position_map.get('Forward', 0)
        
        return f"{defenders}-{midfielders}-{forwards}"
    
    def filter_available_players(self, players: List[Player], 
                                exclude_injured: bool = True,
                                exclude_suspended: bool = True,
                                min_playing_probability: float = MIN_PLAYING_PROBABILITY,
                                max_injury_risk: float = MAX_INJURY_RISK) -> List[Player]:
        """
        Filter players based on availability - ENHANCED
        
        Args:
            players: List of Player objects
            exclude_injured: Whether to exclude injured players
            exclude_suspended: Whether to exclude suspended players
            min_playing_probability: Minimum playing probability threshold
            max_injury_risk: Maximum injury risk threshold
            
        Returns:
            Filtered list of players
        """
        available_players = []
        
        for player in players:
            exclude = False
            
            if exclude_injured and player.is_injured:
                exclude = True
            if exclude_suspended and player.is_suspended:
                exclude = True
            if player.playing_probability < min_playing_probability:
                exclude = True
            if player.injury_risk > max_injury_risk:
                exclude = True
            
            if not exclude:
                available_players.append(player)
        
        return available_players
    
    def calculate_team_value(self, players: List[Player]) -> float:
        """Calculate total team value"""
        return sum(p.cost for p in players)
    
    def calculate_bank_balance(self, players: List[Player]) -> float:
        """Calculate remaining bank balance"""
        total_cost = self.calculate_team_value(players)
        return self.budget - total_cost
    
    def calculate_team_availability_score(self, players: List[Player], 
                                        starting_xi: List[int] = None) -> float:
        """
        Calculate overall team availability score
        
        Args:
            players: List of Player objects
            starting_xi: List of player IDs in starting XI (None means all)
            
        Returns:
            Availability score (0-1)
        """
        if starting_xi:
            relevant_players = [p for p in players if p.player_id in starting_xi]
        else:
            relevant_players = players
        
        if not relevant_players:
            return 0.0
        
        # Weight playing probability by predicted points contribution
        total_weighted_prob = 0.0
        total_weight = 0.0
        
        for player in relevant_players:
            weight = player.predicted_points
            total_weighted_prob += player.playing_probability * weight
            total_weight += weight
        
        return total_weighted_prob / max(total_weight, 0.1)