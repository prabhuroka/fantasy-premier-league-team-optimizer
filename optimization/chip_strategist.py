"""
chip_strategist_enhanced.py - Enhanced chip strategy with single chip recommendation per week
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from optimization.config import WILDCARD_WEEKS, TRIPLE_CAPTAIN_WEEKS
from optimization.constraint_handler import Player


@dataclass
class ChipRecommendation:
    """Chip recommendation data class"""
    chip_name: str
    recommended_gw: int
    expected_gain: float
    confidence: float  # 0-1 confidence score
    reason: str
    alternative_chips: List[str]  # Alternative chips for this week
    risks: List[str]


class ChipStrategist:
    """Enhanced chip strategist that recommends SINGLE best chip per week"""
    
    def __init__(self):
        """Initialize EnhancedChipStrategist"""
        self.wildcard_weeks = WILDCARD_WEEKS
        self.triple_captain_weeks = TRIPLE_CAPTAIN_WEEKS
        
        # Chip priority order (highest to lowest)
        self.chip_priority = ['wildcard', 'triple_captain', 'bench_boost', 'free_hit']
        
        # Minimum expected gain thresholds for each chip
        self.min_gain_thresholds = {
            'wildcard': 15.0,     # Need significant improvement
            'triple_captain': 8.0,  # Good captain with easy fixture/DGW
            'bench_boost': 10.0,   # Strong bench with good fixtures
            'free_hit': 12.0       # Major blank/DGW issues
        }
    
    def recommend_single_chip(self, current_team: List[Player],
                             available_chips: List[str],
                             gameweek: int,
                             fixtures: pd.DataFrame = None,
                             remaining_gws: int = 10) -> Optional[ChipRecommendation]:
        """
        Recommend SINGLE best chip for current gameweek
        
        Args:
            current_team: Current team players
            available_chips: List of available chips
            gameweek: Current gameweek
            fixtures: DataFrame with fixture data
            remaining_gws: Number of remaining gameweeks
            
        Returns:
            Single chip recommendation or None if no chip should be used
        """
        if not available_chips:
            return None
        
        print(f"  Analyzing {len(available_chips)} available chips...")
        
        # Analyze each available chip
        all_recommendations = []
        
        for chip in available_chips:
            if chip == 'wildcard':
                rec = self._recommend_wildcard(current_team, gameweek, remaining_gws, fixtures)
            elif chip == 'triple_captain':
                rec = self._recommend_triple_captain(current_team, gameweek, fixtures)
            elif chip == 'bench_boost':
                rec = self._recommend_bench_boost(current_team, gameweek, fixtures)
            elif chip == 'free_hit':
                rec = self._recommend_free_hit(current_team, gameweek, fixtures)
            else:
                continue
            
            if rec:
                all_recommendations.append(rec)
                print(f"    {chip}: Expected gain = {rec.expected_gain:.1f} points")
        
        if not all_recommendations:
            print(f"    No chips meet minimum gain thresholds")
            return None
        
        # Sort by expected gain (descending)
        all_recommendations.sort(key=lambda x: x.expected_gain, reverse=True)
        
        # Get the best chip
        best_chip = all_recommendations[0]
        
        # Get alternative chips (others with >50% of best chip's gain)
        alternatives = []
        for rec in all_recommendations[1:]:
            if rec.expected_gain >= (best_chip.expected_gain * 0.5):
                alternatives.append(rec.chip_name)
        
        # Create final recommendation
        final_recommendation = ChipRecommendation(
            chip_name=best_chip.chip_name,
            recommended_gw=best_chip.recommended_gw,
            expected_gain=best_chip.expected_gain,
            confidence=best_chip.confidence,
            reason=best_chip.reason,
            alternative_chips=alternatives,
            risks=best_chip.risks
        )
        
        print(f"  → Best chip: {best_chip.chip_name} (expected gain: {best_chip.expected_gain:.1f} points)")
        
        return final_recommendation
    
    def compare_chip_strategies(self, current_team: List[Player],
                              available_chips: List[str],
                              gameweek: int,
                              horizon: int = 5) -> Dict[int, Optional[ChipRecommendation]]:
        """
        Plan chip usage over multiple gameweeks
        
        Args:
            current_team: Current team players
            available_chips: List of available chips
            gameweek: Current gameweek
            horizon: Number of gameweeks to plan for
            
        Returns:
            Dictionary mapping gameweek to chip recommendation
        """
        chip_plan = {}
        remaining_chips = available_chips.copy()
        
        for gw in range(gameweek, min(gameweek + horizon, 39)):
            if not remaining_chips:
                chip_plan[gw] = None
                continue
            
            # Get recommendation for this GW
            recommendation = self.recommend_single_chip(
                current_team=current_team,
                available_chips=remaining_chips,
                gameweek=gw,
                remaining_gws=39 - gw
            )
            
            if recommendation:
                chip_plan[gw] = recommendation
                # Remove used chip from available chips
                if recommendation.chip_name in remaining_chips:
                    remaining_chips.remove(recommendation.chip_name)
                print(f"  GW{gw}: Use {recommendation.chip_name}")
            else:
                chip_plan[gw] = None
                print(f"  GW{gw}: No chip recommended")
        
        return chip_plan
    
    def _recommend_wildcard(self, team: List[Player],
                          current_gw: int,
                          remaining_gws: int,
                          fixtures: pd.DataFrame) -> Optional[ChipRecommendation]:
        """
        Recommend when to use Wildcard chip with gain calculation
        """
        # Check if team needs major rebuild
        team_issues = self._analyze_team_issues(team)
        
        # Calculate team quality score
        team_score = self._calculate_team_score(team)
        
        # Ideal times for wildcard
        ideal_gws = []
        
        # 1. Early wildcard (GW4-8)
        if current_gw <= 8:
            ideal_gws.extend([4, 5, 6, 7, 8])
        
        # 2. January wildcard (GW20-22)
        ideal_gws.extend([20, 21, 22])
        
        # 3. End of season run-in (GW30+)
        if remaining_gws <= 8:
            ideal_gws.append(current_gw + 1)
        
        # Check if team has major issues
        major_issues = sum(1 for issue in team_issues if issue['severity'] == 'high')
        
        expected_gain = 0.0
        reason = ""
        
        if major_issues >= 2 or team_score < 60:
            # Team needs immediate rebuild
            expected_gain = 15.0
            reason = f"Team needs rebuild: {', '.join([i['issue'] for i in team_issues[:2]])}"
            recommended_gw = current_gw + 1
            confidence = 0.8
        elif ideal_gws and current_gw + 1 in ideal_gws:
            # Good strategic time for wildcard
            expected_gain = 10.0
            reason = f"Good strategic time for wildcard (GW{current_gw + 1})"
            recommended_gw = current_gw + 1
            confidence = 0.6
        else:
            return None
        
        # Check minimum gain threshold
        if expected_gain < self.min_gain_thresholds['wildcard']:
            return None
        
        return ChipRecommendation(
            chip_name='wildcard',
            recommended_gw=recommended_gw,
            expected_gain=expected_gain,
            confidence=confidence,
            reason=reason,
            alternative_chips=['triple_captain', 'bench_boost'],  
            risks=['Price changes', 'Injuries', 'Rotation risk']
        )
    
    def _recommend_triple_captain(self, team: List[Player],
                                current_gw: int,
                                fixtures: pd.DataFrame) -> Optional[ChipRecommendation]:
        """
        Recommend when to use Triple Captain chip with gain calculation
        """
        # Look for premium player with easy fixture or DGW
        best_player = None
        best_gain = 0.0
        reason = ""
        
        # Sort players by predicted points
        sorted_team = sorted(team, key=lambda x: x.predicted_points, reverse=True)
        
        # Check top 5 players
        for player in sorted_team[:5]:
            # Calculate normal captain points
            normal_points = player.predicted_points * 2
            
            # Calculate triple captain points
            triple_points = player.predicted_points * 3
            
            # Calculate gain
            gain = triple_points - normal_points
            
            # Bonus for easy fixtures
            if hasattr(player, 'fixture_difficulty') and player.fixture_difficulty <= 2.0:
                gain *= 1.2  # 20% bonus for easy fixtures
            
            # Penalty for rotation risk
            if hasattr(player, 'rotation_risk') and player.rotation_risk > 0.3:
                gain *= 0.8  # 20% penalty for rotation risk
            
            if gain > best_gain:
                best_gain = gain
                best_player = player
                reason = f"{player.name} has high predicted points ({player.predicted_points:.1f})"
                if hasattr(player, 'fixture_difficulty') and player.fixture_difficulty <= 2.0:
                    reason += f" and easy fixture (FDR: {player.fixture_difficulty})"
        
        if best_gain >= self.min_gain_thresholds['triple_captain']:
            return ChipRecommendation(
                chip_name='triple_captain',
                recommended_gw=current_gw + 1,
                expected_gain=best_gain,
                confidence=0.7,
                reason=reason,
                alternative_chips=['bench_boost', 'free_hit'], 
                risks=['Rotation risk', 'Unexpected lineup changes']
            )
        
        return None
    
    def _recommend_bench_boost(self, team: List[Player],
                             current_gw: int,
                             fixtures: pd.DataFrame) -> Optional[ChipRecommendation]:
        """
        Recommend when to use Bench Boost chip with gain calculation
        """
        if len(team) < 15:
            return None
        
        # Sort by predicted points to identify bench
        sorted_team = sorted(team, key=lambda x: x.predicted_points, reverse=True)
        bench = sorted_team[11:15]
        
        # Calculate bench strength
        bench_strength = sum(p.predicted_points for p in bench)
        
        # Only recommend if bench is strong
        if bench_strength >= self.min_gain_thresholds['bench_boost']:
            return ChipRecommendation(
                chip_name='bench_boost',
                recommended_gw=current_gw + 1,
                expected_gain=bench_strength,
                confidence=0.6,
                reason=f"Strong bench expected to score {bench_strength:.1f} points",
                alternative_chips=['triple_captain', 'free_hit'],
                risks=['Unexpected rotation', 'Last-minute injuries']
            )
        
        return None
    
    def _recommend_free_hit(self, team: List[Player],
                          current_gw: int,
                          fixtures: pd.DataFrame) -> Optional[ChipRecommendation]:
        """
        Recommend when to use Free Hit chip with gain calculation
        """
        # Check if many players are missing next GW
        missing_players = []
        for player in team:
            if hasattr(player, 'playing_probability') and player.playing_probability < 0.3:
                missing_players.append(player)
            elif hasattr(player, 'is_injured') and player.is_injured:
                missing_players.append(player)
            elif hasattr(player, 'is_suspended') and player.is_suspended:
                missing_players.append(player)
        
        if len(missing_players) >= 5:  # Significant number missing
            expected_gain = len(missing_players) * 3.0  # Estimate 3 points per replaced player
            
            if expected_gain >= self.min_gain_thresholds['free_hit']:
                return ChipRecommendation(
                    chip_name='free_hit',
                    recommended_gw=current_gw + 1,
                    expected_gain=expected_gain,
                    confidence=0.8,
                    reason=f"{len(missing_players)} players unavailable next gameweek",
                    alternative_chips=['bench_boost', 'wildcard'], 
                    risks=['Unexpected postponements', 'Team selection']
                )
        
        return None
    
    def _analyze_team_issues(self, team: List[Player]) -> List[Dict]:
        """Analyze team for issues that need addressing"""
        issues = []
        
        # Check for injured players
        injured_players = [p for p in team if p.is_injured]
        if injured_players:
            issues.append({
                'issue': f'{len(injured_players)} injured players',
                'severity': 'high' if len(injured_players) >= 2 else 'medium'
            })
        
        # Check for suspended players
        suspended_players = [p for p in team if p.is_suspended]
        if suspended_players:
            issues.append({
                'issue': f'{len(suspended_players)} suspended players',
                'severity': 'high' if len(suspended_players) >= 1 else 'medium'
            })
        
        # Check team value
        team_value = sum(p.cost for p in team)
        if team_value < 95.0:
            issues.append({
                'issue': f'Low team value: £{team_value:.1f}M',
                'severity': 'medium'
            })
        
        return issues
    
    def _calculate_team_score(self, team: List[Player]) -> float:
        """Calculate overall team quality score"""
        if not team:
            return 0.0
        
        # Base score is total predicted points
        base_score = sum(p.predicted_points for p in team)
        
        # Normalize to 0-100 scale
        return (base_score / 15) * 10  # Rough normalization