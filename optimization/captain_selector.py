"""
Captain selection logic
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from optimization.config import CAPTAIN_MULTIPLIER, VICE_CAPTAIN_MULTIPLIER
from optimization.constraint_handler import Player


@dataclass
class CaptainPick:
    """Captain pick data class"""
    player: Player
    predicted_points: float
    captain_points: float  # Points with captain multiplier
    vice_captain_points: float  # Points as vice-captain
    selection_confidence: float  # 0-1 confidence score
    alternative_players: List[Player]  # Alternative options


class CaptainSelector:
    """Select optimal captain and vice-captain"""
    
    def __init__(self):
        """Initialize CaptainSelector"""
        self.captain_multiplier = CAPTAIN_MULTIPLIER
        self.vice_captain_multiplier = VICE_CAPTAIN_MULTIPLIER
        
    def select_captain(self, team: List[Player],
                      consider_fixtures: bool = True,
                      consider_ownership: bool = False,
                      consider_form: bool = True) -> CaptainPick:
        """
        Select optimal captain from team
        
        Args:
            team: List of Player objects in team
            consider_fixtures: Whether to consider fixture difficulty
            consider_ownership: Whether to consider ownership percentage
            consider_form: Whether to consider recent form
            
        Returns:
            CaptainPick object with captain selection
        """
        if not team:
            raise ValueError("Team cannot be empty")
        
        # Calculate captain scores for each player
        captain_scores = []
        
        for player in team:
            base_score = player.predicted_points
            
            # Apply fixture adjustment if enabled
            if consider_fixtures:
                fixture_adjustment = (5.0 - player.fixture_difficulty) / 4.0
                base_score *= (1.0 + 0.1 * fixture_adjustment)  # Bonus for easy fixtures
            
            # Apply ownership adjustment if enabled (differential picks)
            if consider_ownership:
                ownership_factor = (100.0 - player.selected_by_percent) / 100.0
                base_score *= (1.0 + 0.05 * ownership_factor)  # 5% bonus for differentials
            
            # Apply form consideration if enabled
            if consider_form and hasattr(player, 'form'):
                form_factor = min(player.form / 10.0, 2.0)  # Normalize form to 0-2
                base_score *= form_factor
            
            # Reduce score for injury/rotation risk
            risk_penalty = max(player.injury_risk, player.rotation_risk)
            base_score *= (1.0 - risk_penalty)
            
            captain_scores.append({
                'player': player,
                'base_score': base_score,
                'captain_score': base_score * self.captain_multiplier
            })
        
        # Sort by captain score
        captain_scores.sort(key=lambda x: x['captain_score'], reverse=True)
        
        # Get top candidate
        top_candidate = captain_scores[0]
        
        # Get alternative options (next best)
        alternatives = [score['player'] for score in captain_scores[1:4]]
        
        # Calculate confidence score
        if len(captain_scores) > 1:
            score_gap = (top_candidate['captain_score'] - captain_scores[1]['captain_score'])
            max_possible_gap = top_candidate['captain_score'] * 0.5  # Arbitrary max gap
            confidence = min(score_gap / max_possible_gap, 1.0)
        else:
            confidence = 1.0
        
        return CaptainPick(
            player=top_candidate['player'],
            predicted_points=top_candidate['base_score'],
            captain_points=top_candidate['captain_score'],
            vice_captain_points=top_candidate['base_score'],
            selection_confidence=confidence,
            alternative_players=alternatives
        )
    
    def select_vice_captain(self, team: List[Player],
                          captain_id: int) -> CaptainPick:
        """
        Select vice-captain (best option after captain)
        
        Args:
            team: List of Player objects in team
            captain_id: ID of selected captain
            
        Returns:
            CaptainPick object for vice-captain
        """
        # Filter out captain
        eligible_players = [p for p in team if p.player_id != captain_id]
        
        if not eligible_players:
            # If no other players, return captain as vice (shouldn't happen)
            captain = next((p for p in team if p.player_id == captain_id), None)
            if captain:
                return CaptainPick(
                    player=captain,
                    predicted_points=captain.predicted_points,
                    captain_points=captain.predicted_points * self.captain_multiplier,
                    vice_captain_points=captain.predicted_points,
                    selection_confidence=1.0,
                    alternative_players=[]
                )
            else:
                raise ValueError("No eligible players for vice-captain")
        
        # Select best remaining player
        vice_candidates = sorted(eligible_players, 
                                key=lambda x: x.predicted_points, 
                                reverse=True)
        
        top_vice = vice_candidates[0]
        alternatives = vice_candidates[1:3]
        
        return CaptainPick(
            player=top_vice,
            predicted_points=top_vice.predicted_points,
            captain_points=top_vice.predicted_points * self.captain_multiplier,
            vice_captain_points=top_vice.predicted_points,
            selection_confidence=0.8,  # Lower confidence for vice
            alternative_players=alternatives
        )
    
    def analyze_captain_options(self, team: List[Player],
                               top_n: int = 5) -> List[Dict]:
        """
        Analyze top captain options
        
        Args:
            team: List of Player objects
            top_n: Number of top options to return
            
        Returns:
            List of captain option dictionaries
        """
        captain_options = []
        
        for player in team:
            # Calculate captain score
            captain_score = player.predicted_points * self.captain_multiplier
            
            # Apply adjustments
            fixture_bonus = (5.0 - player.fixture_difficulty) / 10.0  # 0-0.5 bonus
            captain_score *= (1.0 + fixture_bonus)
            
            # Risk adjustment
            risk_penalty = max(player.injury_risk, player.rotation_risk)
            captain_score *= (1.0 - risk_penalty)
            
            captain_options.append({
                'player_id': player.player_id,
                'player_name': player.name,
                'position': player.position,
                'team_name': player.team_name,
                'predicted_points': player.predicted_points,
                'captain_score': captain_score,
                'fixture_difficulty': player.fixture_difficulty,
                'injury_risk': player.injury_risk,
                'rotation_risk': player.rotation_risk,
                'selected_by_percent': player.selected_by_percent
            })
        
        # Sort and return top N
        captain_options.sort(key=lambda x: x['captain_score'], reverse=True)
        return captain_options[:top_n]
    
    def recommend_captain_chip(self, team: List[Player],
                             chip_available: bool = True,
                             double_gw_players: List[Player] = None) -> Dict:
        """
        Recommend when to use captain chip (Triple Captain)
        
        Args:
            team: List of Player objects
            chip_available: Whether chip is available
            double_gw_players: Players with double gameweeks
            
        Returns:
            Chip recommendation
        """
        if not chip_available:
            return {
                'recommendation': 'do_not_use',
                'reason': 'Chip not available',
                'expected_gain': 0.0
            }
        
        # Check for double gameweek players
        if double_gw_players:
            # Find best double gameweek player in team
            dgw_in_team = [p for p in team if p.player_id in [dp.player_id for dp in double_gw_players]]
            
            if dgw_in_team:
                best_dgw = max(dgw_in_team, key=lambda x: x.predicted_points)
                
                # Calculate expected gain from triple captain
                normal_captain_points = best_dgw.predicted_points * 2
                triple_captain_points = best_dgw.predicted_points * 3
                expected_gain = triple_captain_points - normal_captain_points
                
                return {
                    'recommendation': 'use_triple_captain',
                    'player_id': best_dgw.player_id,
                    'player_name': best_dgw.name,
                    'expected_points_normal': normal_captain_points,
                    'expected_points_triple': triple_captain_points,
                    'expected_gain': expected_gain,
                    'reason': f'{best_dgw.name} has double gameweek'
                }
        
        # Check for premium player with easy fixture
        easy_fixture_players = [p for p in team if p.fixture_difficulty <= 2.0]
        
        if easy_fixture_players:
            best_easy_fixture = max(easy_fixture_players, key=lambda x: x.predicted_points)
            
            if best_easy_fixture.predicted_points >= 8.0:  # High predicted points
                normal_captain_points = best_easy_fixture.predicted_points * 2
                triple_captain_points = best_easy_fixture.predicted_points * 3
                expected_gain = triple_captain_points - normal_captain_points
                
                return {
                    'recommendation': 'consider_triple_captain',
                    'player_id': best_easy_fixture.player_id,
                    'player_name': best_easy_fixture.name,
                    'expected_points_normal': normal_captain_points,
                    'expected_points_triple': triple_captain_points,
                    'expected_gain': expected_gain,
                    'reason': f'{best_easy_fixture.name} has easy fixture and high predicted points'
                }
        
        # Default recommendation
        return {
            'recommendation': 'save_for_later',
            'reason': 'No compelling opportunity found',
            'expected_gain': 0.0
        }