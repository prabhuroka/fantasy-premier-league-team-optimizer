"""
Calculate fixture difficulty ratings
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math

class FixtureDifficultyCalculator:
    def __init__(self, db=None):
        """
        Initialize FixtureDifficultyCalculator with database connection.
        
        Args:
            db: FPLDatabase instance for data access
        """
        self.db = db
        self.base_elo = 1500  
        self.k_factor = 32   
        self.home_advantage = 50 
        
    def calculate_team_elo(self, team_id: int, gameweek: int, 
                          season: str = "2025-2026") -> float:
        """
        Get team's Elo rating at specific gameweek using linear interpolation.
        
        Args:
            team_id: Team ID
            gameweek: Gameweek number
            season: Season string
            
        Returns:
            Elo rating at specified gameweek
        """
        try:
            if self.db:
                # Try to get Elo from database
                query = """
                    SELECT elo 
                    FROM teams 
                    WHERE team_id = ? AND season = ?
                """
                team_elos = self.db.execute_query(query, (team_id, season))
                
                if team_elos is not None and not team_elos.empty:
                    # The teams table has elo but no gameweek, so just return the elo value
                    return float(team_elos.iloc[0]['elo'])
            # Default Elo if not found
            return self.base_elo
            
        except Exception as e:
            print(f"Error calculating team Elo: {e}")
            return self.base_elo
    
    def calculate_fdr(self, team_id: int, opponent_id: int, 
                     is_home: bool, gameweek: int, 
                     season: str = "2025-2026") -> float:
        """
        Calculate Fixture Difficulty Rating (1-5 scale).
        
        Formula: FDR = 3 + ((opponent_elo - team_elo) / 200) + home_advantage
        
        Args:
            team_id: Team ID
            opponent_id: Opponent team ID
            is_home: Whether team is playing at home
            gameweek: Gameweek number
            season: Season string
            
        Returns:
            FDR score (1.0-5.0)
        """
        try:
            # Get Elo ratings
            team_elo = self.calculate_team_elo(team_id, gameweek, season)
            opponent_elo = self.calculate_team_elo(opponent_id, gameweek, season)
            
            # Calculate base FDR
            elo_difference = opponent_elo - team_elo
            home_advantage_factor = -0.5 if is_home else 0  # Easier at home
            
            fdr = 3.0 + (elo_difference / 200.0) + home_advantage_factor
            
            # Clamp to 1-5 range
            fdr = max(1.0, min(5.0, fdr))
            
            return round(fdr, 2)
            
        except Exception as e:
            print(f"Error calculating FDR: {e}")
            return 3.0  
    
    def get_fixture_run_difficulty(self, team_id: int, start_gw: int, 
                                  length: int = 5, season: str = "2025-2026") -> List[float]:
        """
        Calculate difficulty of next N fixtures for a team.
        
        Args:
            team_id: Team ID
            start_gw: Starting gameweek
            length: Number of fixtures to analyze
            season: Season string
            
        Returns:
            List of FDR scores for each fixture
        """
        try:
            if self.db:
                query = """
                    SELECT gameweek, opponent_team_id, was_home
                    FROM player_gameweek_stats 
                    WHERE team_id = ? AND season = ? AND gameweek >= ?
                    GROUP BY gameweek, opponent_team_id, was_home
                    ORDER BY gameweek
                    LIMIT ?
                """
                fixtures = self.db.execute_query(query, (team_id, season, start_gw, length))
                
                fdr_scores = []
                
                if fixtures is not None and not fixtures.empty:
                    for _, row in fixtures.iterrows():
                        fixture_gw = row['gameweek']
                        opponent_id = row['opponent_team_id']
                        was_home = row['was_home']
                        fdr = self.calculate_fdr(team_id, opponent_id, was_home, fixture_gw, season)
                        fdr_scores.append(fdr)
                
                # If fewer fixtures than requested, fill with neutral difficulty
                while len(fdr_scores) < length:
                    fdr_scores.append(3.0)
                
                return fdr_scores[:length]
            
            # Default if no database
            return [3.0] * length
            
        except Exception as e:
            print(f"Error getting fixture run difficulty: {e}")
            return [3.0] * length
    
    def calculate_defensive_strength(self, team_id: int, gameweek: int, 
                                    is_home: bool = True, 
                                    season: str = "2025-2026",
                                    lookback_gws: int = 5) -> float:
        """
        Calculate team's defensive strength.
        
        Args:
            team_id: Team ID
            gameweek: Current gameweek
            is_home: Filter for home/away games
            season: Season string
            lookback_gws: Number of gameweeks to look back
            
        Returns:
            Defensive strength score (0-1, higher is better defense)
        """
        try:
            if self.db:
                # Get defensive stats for the team
                query = """
                    SELECT 
                        AVG(CASE WHEN goals_conceded IS NULL THEN 0 ELSE goals_conceded END) as avg_goals_conceded,
                        AVG(CASE WHEN clean_sheet IS NULL THEN 0 ELSE clean_sheet END) as clean_sheet_rate,
                        COUNT(*) as games_played
                    FROM player_gameweek_stats 
                    WHERE team_id = ? 
                      AND opponent_team_id = ?  -- Filter by opponent being the team
                      AND gameweek <= ? 
                      AND gameweek > ? - ?
                      AND season = ?
                      AND was_home = ?
                """
                
                # For defensive strength, we look at games where this team was the opponent
                stats = self.db.execute_query(query, (team_id, team_id, gameweek, gameweek, lookback_gws, season, not is_home))
                
                if stats is not None and not stats.empty:
                    row = stats.iloc[0]
                    games_played = row['games_played']
                
                    if games_played > 0:  # games_played > 0
                        avg_goals_conceded = row['avg_goals_conceded'] if pd.notna(row['avg_goals_conceded']) else 0
                        clean_sheet_rate = row['clean_sheet_rate'] if pd.notna(row['clean_sheet_rate']) else 0
                    
                        # Convert to defensive strength (0-1)
                        # Lower goals conceded and higher clean sheet rate = better defense
                        goals_strength = max(0, 1 - (avg_goals_conceded / 3))  # Assume 3+ goals is worst
                        cs_strength = clean_sheet_rate
                    
                        # Weighted average
                        defensive_strength = (goals_strength * 0.6) + (cs_strength * 0.4)
                    
                        return round(max(0, min(1, defensive_strength)), 3)
            
            return 0.5  # Default neutral strength
            
        except Exception as e:
            print(f"Error calculating defensive strength: {e}")
            return 0.5
    
    def calculate_attacking_strength(self, team_id: int, gameweek: int, 
                                    is_home: bool = True, 
                                    season: str = "2025-2026",
                                    lookback_gws: int = 5) -> float:
        """
        Calculate team's attacking strength.
        
        Args:
            team_id: Team ID
            gameweek: Current gameweek
            is_home: Filter for home/away games
            season: Season string
            lookback_gws: Number of gameweeks to look back
            
        Returns:
            Attacking strength score (0-1, higher is better attack)
        """
        try:
            if self.db:
                # Get attacking stats for the team
                query = """
                    SELECT 
                        AVG(CASE WHEN goals_scored IS NULL THEN 0 ELSE goals_scored END) as avg_goals_scored,
                        AVG(CASE WHEN expected_goals_xg IS NULL THEN 0 ELSE expected_goals_xg END) as avg_xg,
                        COUNT(*) as games_played
                    FROM player_gameweek_stats 
                    WHERE team_id = ? 
                      AND gameweek <= ? 
                      AND gameweek > ? - ?
                      AND season = ?
                      AND was_home = ?
                    GROUP BY gameweek
                """
                
                stats = self.db.execute_query(query, (team_id, gameweek, gameweek, lookback_gws, season, is_home))
                
                if stats is not None and not stats.empty:
                    # Check if we have any valid rows
                        valid_stats = stats[(stats['avg_goals_scored'].notna()) & (stats['avg_xg'].notna())]
                        
                        if not valid_stats.empty:
                            # Average across all gameweeks
                            avg_goals = valid_stats['avg_goals_scored'].mean()
                            avg_xg = valid_stats['avg_xg'].mean()
                    
                            # Convert to attacking strength (0-1)
                            # Higher goals and xG = better attack
                            goals_strength = min(1, avg_goals / 2.5)  # Assume 2.5+ goals is best
                            xg_strength = min(1, avg_xg / 2.0)  # Assume 2.0+ xG is best
                    
                            # Weighted average
                            attacking_strength = (goals_strength * 0.7) + (xg_strength * 0.3)
                    
                            return round(max(0, min(1, attacking_strength)), 3)
        
            return 0.5  # Default neutral strength    
            
        except Exception as e:
            print(f"Error calculating attacking strength: {e}")
            return 0.5
    
    def calculate_fixture_difficulty_matrix(self, gameweek: int, 
                                          season: str = "2025-2026") -> pd.DataFrame:
        """
        Calculate FDR matrix for all teams in a gameweek.
        
        Args:
            gameweek: Gameweek number
            season: Season string
            
        Returns:
            DataFrame with FDR matrix
        """
        try:
            if self.db:
                # Get all teams
                query = "SELECT DISTINCT team_id FROM teams WHERE season = ?"
                teams = self.db.execute_query(query, (season,))
                
                if not teams:
                    return pd.DataFrame()
                
                team_ids = [t[0] for t in teams]
                
                # Get fixtures for the gameweek
                fixtures_query = """
                    SELECT DISTINCT team_id, opponent_team_id, was_home
                    FROM player_gameweek_stats 
                    WHERE gameweek = ? AND season = ?
                """
                fixtures = self.db.execute_query(fixtures_query, (gameweek, season))
                
                # Create matrix
                matrix_data = []
                
                for team_id in team_ids:
                    # Find team's fixture
                    team_fixture = next((f for f in fixtures if f[0] == team_id), None)
                    
                    if team_fixture:
                        _, opponent_id, was_home = team_fixture
                        fdr = self.calculate_fdr(team_id, opponent_id, was_home, gameweek, season)
                        
                        matrix_data.append({
                            'team_id': team_id,
                            'gameweek': gameweek,
                            'opponent_id': opponent_id,
                            'is_home': was_home,
                            'fdr': fdr,
                            'team_elo': self.calculate_team_elo(team_id, gameweek, season),
                            'opponent_elo': self.calculate_team_elo(opponent_id, gameweek, season)
                        })
                
                return pd.DataFrame(matrix_data)
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error calculating fixture matrix: {e}")
            return pd.DataFrame()
    
    def get_team_form(self, team_id: int, gameweek: int, 
                     season: str = "2025-2026", 
                     form_gws: int = 5) -> Dict[str, float]:
        """
        Calculate team's recent form metrics.
        
        Args:
            team_id: Team ID
            gameweek: Current gameweek
            season: Season string
            form_gws: Number of gameweeks to consider for form
            
        Returns:
            Dictionary with form metrics
        """
        try:
            if self.db:
                # Get team's recent results
                query = """
                    SELECT 
                        AVG(total_points) as avg_team_points,
                        AVG(goals_scored) as avg_goals_for,
                        AVG(goals_conceded) as avg_goals_against,
                        SUM(CASE WHEN total_points > 0 THEN 1 ELSE 0 END) / COUNT(*) as win_rate
                    FROM (
                        SELECT DISTINCT gameweek, team_id, 
                               SUM(total_points) as total_points,
                               SUM(goals_scored) as goals_scored,
                               SUM(goals_conceded) as goals_conceded
                        FROM player_gameweek_stats 
                        WHERE team_id = ? 
                          AND gameweek <= ? 
                          AND gameweek > ? - ?
                          AND season = ?
                        GROUP BY gameweek, team_id
                    )
                """
                
                stats = self.db.execute_query(query, (team_id, gameweek, gameweek, form_gws, season))
                
                if stats is not None and not stats.empty:
                    row = stats.iloc[0]
                    avg_points = row['avg_team_points'] if pd.notna(row['avg_team_points']) else 0
                    avg_goals_for = row['avg_goals_for'] if pd.notna(row['avg_goals_for']) else 0
                    avg_goals_against = row['avg_goals_against'] if pd.notna(row['avg_goals_against']) else 0
                    win_rate = row['win_rate'] if pd.notna(row['win_rate']) else 0
                
                    return {
                        'avg_points': float(avg_points),
                        'avg_goals_for': float(avg_goals_for),
                        'avg_goals_against': float(avg_goals_against),
                        'win_rate': float(win_rate),
                        'goal_difference': float(avg_goals_for - avg_goals_against)
                    }
            
            return {
                'avg_points': 0,
                'avg_goals_for': 0,
                'avg_goals_against': 0,
                'win_rate': 0,
                'goal_difference': 0
            }
            
        except Exception as e:
            print(f"Error getting team form: {e}")
            return {
                'avg_points': 0,
                'avg_goals_for': 0,
                'avg_goals_against': 0,
                'win_rate': 0,
                'goal_difference': 0
            }
    
    def update_elo_ratings(self, match_results: List[Dict], 
                          season: str = "2025-2026") -> Dict[int, float]:
        """
        Update Elo ratings based on match results.
        
        Args:
            match_results: List of match result dictionaries
            season: Season string
            
        Returns:
            Dictionary of updated Elo ratings
        """
        try:
            updated_elos = {}
            
            for match in match_results:
                home_team = match.get('home_team_id')
                away_team = match.get('away_team_id')
                home_score = match.get('home_score', 0)
                away_score = match.get('away_score', 0)
                gameweek = match.get('gameweek', 1)
                
                if not home_team or not away_team:
                    continue
                
                # Get current Elo ratings
                home_elo = self.calculate_team_elo(home_team, gameweek, season)
                away_elo = self.calculate_team_elo(away_team, gameweek, season)
                
                # Apply home advantage
                home_elo_adj = home_elo + self.home_advantage
                
                # Calculate expected scores
                home_expected = 1 / (1 + 10 ** ((away_elo - home_elo_adj) / 400))
                away_expected = 1 - home_expected
                
                # Determine actual score (1 for win, 0.5 for draw, 0 for loss)
                if home_score > away_score:
                    home_actual = 1
                    away_actual = 0
                elif home_score < away_score:
                    home_actual = 0
                    away_actual = 1
                else:
                    home_actual = 0.5
                    away_actual = 0.5
                
                # Update Elo ratings
                home_new_elo = home_elo + self.k_factor * (home_actual - home_expected)
                away_new_elo = away_elo + self.k_factor * (away_actual - away_expected)
                
                updated_elos[home_team] = home_new_elo
                updated_elos[away_team] = away_new_elo
            
            return updated_elos
            
        except Exception as e:
            print(f"Error updating Elo ratings: {e}")
            return {}