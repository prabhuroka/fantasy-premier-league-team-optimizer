"""
Process injury data and calculate injury risk
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
from collections import defaultdict

class InjuryProcessor:
    def __init__(self, db=None):
        """
        Initialize InjuryProcessor with database connection.
        
        Args:
            db: FPLDatabase instance for data access
        """
        self.db = db
        
        # Injury type patterns and typical recovery times (in days)
        self.injury_patterns = {
            'hamstring': {
                'keywords': ['hamstring', 'ham'],
                'severity': {
                    'minor': 7,
                    'moderate': 14,
                    'major': 28,
                    'severe': 42
                }
            },
            'knee': {
                'keywords': ['knee', 'acl', 'mcl', 'pcl', 'ligament'],
                'severity': {
                    'minor': 10,
                    'moderate': 21,
                    'major': 56,
                    'severe': 180
                }
            },
            'ankle': {
                'keywords': ['ankle', 'sprain'],
                'severity': {
                    'minor': 7,
                    'moderate': 14,
                    'major': 28,
                    'severe': 60
                }
            },
            'muscle': {
                'keywords': ['muscle', 'strain', 'tear'],
                'severity': {
                    'minor': 7,
                    'moderate': 21,
                    'major': 42,
                    'severe': 90
                }
            },
            'calf': {
                'keywords': ['calf'],
                'severity': {
                    'minor': 10,
                    'moderate': 21,
                    'major': 35,
                    'severe': 60
                }
            },
            'groin': {
                'keywords': ['groin', 'adductor'],
                'severity': {
                    'minor': 7,
                    'moderate': 14,
                    'major': 28,
                    'severe': 45
                }
            },
            'head': {
                'keywords': ['head', 'concussion'],
                'severity': {
                    'minor': 7,
                    'moderate': 14,
                    'major': 21,
                    'severe': 30
                }
            },
            'shoulder': {
                'keywords': ['shoulder', 'collarbone', 'clavicle'],
                'severity': {
                    'minor': 14,
                    'moderate': 28,
                    'major': 56,
                    'severe': 90
                }
            },
            'illness': {
                'keywords': ['ill', 'virus', 'sick', 'infection'],
                'severity': {
                    'minor': 3,
                    'moderate': 7,
                    'major': 14,
                    'severe': 21
                }
            },
            'knock': {
                'keywords': ['knock', 'bruise'],
                'severity': {
                    'minor': 3,
                    'moderate': 7,
                    'major': 14,
                    'severe': 21
                }
            }
        }
        
        # Training status patterns
        self.training_patterns = {
            'full_training': ['full training', 'training normally', 'back in training'],
            'light_training': ['light training', 'modified training', 'individual training'],
            'no_training': ['not training', 'still out', 'continues to miss'],
            'recovering': ['recovering', 'rehabilitation', 'rehab']
        }
    
    def parse_injury_news(self, news_text: str) -> Dict[str, Any]:
        """
        Parse injury news text to extract structured information.
        
        Args:
            news_text: Injury news text from FPL
            
        Returns:
            Dictionary with parsed injury information
        """
        if not news_text or str(news_text).lower() in ['nan', 'none', '']:
            return {
                'injury_type': None,
                'severity': None,
                'expected_return_days': None,
                'training_status': 'full_training',
                'confidence': 0.0
            }
        
        news_lower = news_text.lower()
        result = {
            'injury_type': None,
            'severity': 'unknown',
            'expected_return_days': None,
            'training_status': 'unknown',
            'confidence': 0.0
        }
        
        # Check for common phrases indicating no injury
        no_injury_phrases = [
            'available', 'fit', 'ready', 'no issues', 'fully fit',
            'expected to be available', 'should be available'
        ]
        
        for phrase in no_injury_phrases:
            if phrase in news_lower:
                return {
                    'injury_type': None,
                    'severity': None,
                    'expected_return_days': 0,
                    'training_status': 'full_training',
                    'confidence': 0.9
                }
        
        # Identify injury type
        for injury_type, pattern in self.injury_patterns.items():
            for keyword in pattern['keywords']:
                if keyword in news_lower:
                    result['injury_type'] = injury_type
                    result['confidence'] += 0.3
                    break
            
            if result['injury_type']:
                break
        
        # Identify severity
        severity_keywords = {
            'minor': ['minor', 'small', 'slight', 'knock', 'bruise'],
            'moderate': ['moderate', 'medium', 'some', 'few'],
            'major': ['major', 'significant', 'serious', 'severe'],
            'long-term': ['long-term', 'long term', 'months', 'surgery']
        }
        
        for severity, keywords in severity_keywords.items():
            for keyword in keywords:
                if keyword in news_lower:
                    result['severity'] = severity
                    result['confidence'] += 0.2
                    break
        
        # Extract expected return timeframe
        time_patterns = [
            (r'(\d+)\s*day', 1),          # X days
            (r'(\d+)\s*week', 7),         # X weeks
            (r'(\d+)\s*month', 30),       # X months
            (r'next\s*gameweek', 7),      # Next gameweek
            (r'back\s*in\s*(\d+)', 1),    # Back in X (days)
            (r'(\d+)-(\d+)\s*day', 1),    # X-Y days
            (r'(\d+)-(\d+)\s*week', 7),   # X-Y weeks
        ]
        
        for pattern, multiplier in time_patterns:
            match = re.search(pattern, news_lower)
            if match:
                try:
                    if len(match.groups()) == 2:
                        # Range like "2-3 weeks"
                        days = (int(match.group(1)) + int(match.group(2))) / 2 * multiplier
                    else:
                        # Single number
                        days = int(match.group(1)) * multiplier
                    
                    result['expected_return_days'] = int(days)
                    result['confidence'] += 0.2
                    break
                except:
                    continue
        
        # If no specific timeframe found, estimate based on severity
        if result['expected_return_days'] is None and result['severity'] != 'unknown':
            if result['injury_type'] in self.injury_patterns:
                typical_days = self.injury_patterns[result['injury_type']]['severity'].get(result['severity'], 14)
                result['expected_return_days'] = typical_days
        
        # Determine training status
        for status, patterns in self.training_patterns.items():
            for pattern in patterns:
                if pattern in news_lower:
                    result['training_status'] = status
                    result['confidence'] += 0.1
                    break
        
        # Cap confidence at 1.0
        result['confidence'] = min(1.0, result['confidence'])
        
        return result
    
    def calculate_injury_risk(self, player_id: int, 
                         injury_history: pd.DataFrame = None) -> float:
        """
        Calculate injury risk probability (0-1) for a player.
    
        Args:
            player_id: Player ID
            injury_history: DataFrame with injury history (optional)
        
        Returns:
            Injury risk score (0.0-1.0)
        """
        try:
            risk_score = 0.0
        
            # If no injury history provided, try to get from database
            if injury_history is None and self.db:
                # Get recent injury news for player
                try:
                    query = """
                        SELECT news, chance_of_playing_next_round
                        FROM players 
                        WHERE player_id = ? 
                        AND season = '2025-2026'
                        LIMIT 1
                    """
                    injury_data = self.db.execute_query(query, (player_id,))
                
                    if injury_data is not None:
                        # Handle DataFrame
                        if isinstance(injury_data, pd.DataFrame) and not injury_data.empty:
                            latest_news = injury_data.iloc[0]['news']
                            chance_playing = injury_data.iloc[0]['chance_of_playing_next_round']
                        # Handle list
                        elif isinstance(injury_data, list) and injury_data:
                            latest_news = injury_data[0][0] if injury_data[0][0] else ""
                            chance_playing = injury_data[0][1] if injury_data[0][1] else 100
                        # Calculate risk from chance of playing
                        if chance_playing is not None:
                            risk_from_chance = (100 - chance_playing) / 100
                            risk_score += risk_from_chance * 0.6
                    
                        # Parse injury news
                        if latest_news:
                            parsed_news = self.parse_injury_news(latest_news)
                        
                            # Add risk based on injury details
                            if parsed_news['injury_type']:
                                risk_score += 0.3 * parsed_news['confidence']
                        
                            if parsed_news['severity'] in ['major', 'long-term']:
                                risk_score += 0.2
                            elif parsed_news['severity'] == 'moderate':
                                risk_score += 0.1
                        
                            if parsed_news['training_status'] == 'no_training':
                                risk_score += 0.2
                            elif parsed_news['training_status'] == 'light_training':
                                risk_score += 0.1
                except Exception as e:
                    print(f"    Error querying injury data for player {player_id}: {e}")
                    # Use default risk
                    pass
        
            # Add base risk based on position (some positions are more injury-prone)
            position_risk = 0.05  # Base 5% risk
        
            risk_score = min(1.0, risk_score + position_risk)
        
            return round(risk_score, 3)
        
        except Exception as e:
            print(f"Error calculating injury risk: {e}")
            return 0.1  # Default low risk
    
    def estimate_return_date(self, injury_type: str, 
                            severity: str, 
                            current_date: datetime = None) -> datetime:
        """
        Estimate return date based on injury type and severity.
        
        Args:
            injury_type: Type of injury
            severity: Injury severity
            current_date: Current date (defaults to now)
            
        Returns:
            Estimated return datetime
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Default recovery time (2 weeks)
        recovery_days = 14
        
        # Look up typical recovery time
        if injury_type in self.injury_patterns:
            if severity in self.injury_patterns[injury_type]['severity']:
                recovery_days = self.injury_patterns[injury_type]['severity'][severity]
        
        # Add some uncertainty (±20%)
        uncertainty = recovery_days * 0.2
        adjusted_days = recovery_days + np.random.uniform(-uncertainty, uncertainty)
        
        return current_date + timedelta(days=adjusted_days)
    
    def get_playing_probability(self, player_id: int, 
                               gameweek: int) -> float:
        """
        Calculate probability player will play in specified gameweek.
        
        Args:
            player_id: Player ID
            gameweek: Target gameweek
            
        Returns:
            Playing probability (0.0-1.0)
        """
        try:
            if self.db:
                # Get player's current status
                query = """
                    SELECT news, chance_of_playing_next_round
                    FROM players 
                    WHERE player_id = ? 
                    AND season = '2025-2026'
                    LIMIT 1
                """
                player_data = self.db.execute_query(query, (player_id,))
                
                if player_data is not None and not player_data.empty:
                    row = player_data.iloc[0]
                    news = row['news'] if 'news' in row else ""
                    chance_playing = row['chance_of_playing_next_round'] if 'chance_of_playing_next_round' in row else None    
                    
                    # If chance_playing is provided, use it directly
                    if chance_playing is not None:
                        base_probability = chance_playing / 100.0
                    else:
                        # Parse injury news
                        parsed_news = self.parse_injury_news(news if news else "")
                        
                        # Calculate probability based on parsed news
                        if parsed_news['injury_type'] is None:
                            base_probability = 0.95  # No injury news = high probability
                        else:
                            # Adjust based on severity
                            if parsed_news['severity'] == 'minor':
                                base_probability = 0.7
                            elif parsed_news['severity'] == 'moderate':
                                base_probability = 0.4
                            elif parsed_news['severity'] in ['major', 'long-term']:
                                base_probability = 0.1
                            else:
                                base_probability = 0.5
                            
                            # Adjust based on training status
                            if parsed_news['training_status'] == 'full_training':
                                base_probability *= 1.2
                            elif parsed_news['training_status'] == 'light_training':
                                base_probability *= 0.8
                            elif parsed_news['training_status'] == 'no_training':
                                base_probability *= 0.3
                    
                    # Adjust for gameweek (if injury happened recently)
                    # This is a simplified model
                    current_gw = self.get_current_gameweek()
                    
                    if current_gw:
                        gw_difference = gameweek - current_gw
                        
                        if gw_difference <= 0:
                            # Past or current gameweek
                            return min(1.0, base_probability)
                        else:
                            # Future gameweek - probability increases with time
                            recovery_factor = min(1.0, gw_difference * 0.2)
                            adjusted_probability = min(1.0, base_probability + (1 - base_probability) * recovery_factor)
                            return adjusted_probability
                    
                    return min(1.0, base_probability)
            
            # Default if no data
            return 0.8
            
        except Exception as e:
            print(f"Error getting playing probability: {e}")
            return 0.5  # Default 50% probability
    
    def get_current_gameweek(self) -> Optional[int]:
        """
        Get current gameweek number.
        
        Returns:
            Current gameweek number or None if not available
        """
        try:
            if self.db:
                query = "SELECT MAX(gameweek) FROM player_gameweek_stats"
                result = self.db.execute_query(query)
                if result and result[0][0]:
                    return result[0][0]
            
            # Estimate based on current date (for 2025-2026 season)
            season_start = datetime(2025, 8, 9)  # Approximate start
            current_date = datetime.now()
            
            if current_date < season_start:
                return 1
            
            weeks_passed = (current_date - season_start).days // 7
            return min(38, max(1, weeks_passed + 1))
            
        except:
            return None
    
    def analyze_injury_history(self,player_id: int,
                               lookback_days: int = 365) -> Dict[str, Any]:
        """
        Analyze player's injury history.

        Args:
            player_id: Player ID
            lookback_days: Number of days to look back

        Returns:
            Dictionary with injury history analysis
        """
        try:
            if self.db:
                # Get injury history from player news
                try:
                    query = """
                        SELECT news
                        FROM players 
                        WHERE player_id = ? 
                            AND news IS NOT NULL 
                            AND news != ''
                        LIMIT 10
                    """
                    injury_history = self.db.execute_query(query, (player_id,))

                    if injury_history is None or injury_history.empty:
                        return {
                            'total_injuries': 0,
                            'recent_injuries': 0,
                            'avg_recovery_time': 0,
                            'injury_prone_score': 0.0,
                            'most_common_injury': None
                            }

                    total_injuries = 0
                    recent_injuries = 0
                    injury_types = defaultdict(int)

                    for _, row in injury_history.iterrows():
                        news = row['news']
                        if news:
                            parsed = self.parse_injury_news(news)

                            if parsed['injury_type']:
                                total_injuries += 1
                                recent_injuries += 1
                                injury_types[parsed['injury_type']] += 1

                    injury_prone_score = min(1.0,(recent_injuries * 0.3) + (total_injuries * 0.1))

                    most_common = None
                    if injury_types:
                        most_common = max(injury_types.items(), key=lambda x: x[1])[0]

                    return {
                        'total_injuries': total_injuries,
                        'recent_injuries': recent_injuries,
                        'avg_recovery_time': 14,
                        'injury_prone_score': round(injury_prone_score, 3),
                        'most_common_injury': most_common
                        }

                except Exception as e:
                    print(f"    Error analyzing injury history: {e}")
                    return {
                        'total_injuries': 0,
                        'recent_injuries': 0,
                        'avg_recovery_time': 0,
                        'injury_prone_score': 0.0,
                        'most_common_injury': None
                        }
                    
            return {
                'total_injuries': 0,
                'recent_injuries': 0,
                'avg_recovery_time': 0,
                'injury_prone_score': 0.0,
                'most_common_injury': None
                }
        
        except Exception as e:
            print(f"Error analyzing injury history: {e}")
            return {
                'total_injuries': 0,
                'recent_injuries': 0,
                'avg_recovery_time': 0,
                'injury_prone_score': 0.0,
                'most_common_injury': None
                }


    def get_injury_report(self, team_id: int = None, 
                         severity: str = None) -> pd.DataFrame:
        """
        Generate injury report for players/teams.
        
        Args:
            team_id: Optional team ID to filter
            severity: Optional severity filter
            
        Returns:
            DataFrame with injury report
        """
        try:
            if self.db:
                # Build query
                query = """
                    SELECT p.player_id, p.web_name, p.team_id, p.position,
                           p.news, p.chance_of_playing_next_round, p.updated_at
                    FROM players p
                    WHERE p.news IS NOT NULL 
                      AND p.news != ''
                """
                
                params = []
                
                if team_id:
                    query += " AND p.team_id = ?"
                    params.append(team_id)
                
                query += " ORDER BY p.chance_of_playing_next_round ASC, p.updated_at DESC"
                
                injury_data = self.db.execute_query(query, tuple(params))
                
                if not injury_data:
                    return pd.DataFrame()
                
                # Process results
                report_rows = []
                
                for row in injury_data:
                    player_id, web_name, team_id, position, news, chance_playing, updated_at = row
                    
                    parsed_news = self.parse_injury_news(news)
                    
                    # Apply severity filter
                    if severity and parsed_news['severity'] != severity:
                        continue
                    
                    playing_prob = self.get_playing_probability(player_id, self.get_current_gameweek() or 1)
                    
                    report_rows.append({
                        'player_id': player_id,
                        'player_name': web_name,
                        'team_id': team_id,
                        'position': position,
                        'injury_type': parsed_news['injury_type'],
                        'severity': parsed_news['severity'],
                        'training_status': parsed_news['training_status'],
                        'expected_return_days': parsed_news['expected_return_days'],
                        'chance_of_playing': chance_playing,
                        'playing_probability': round(playing_prob, 3),
                        'news': news,
                        'last_updated': updated_at
                    })
                
                return pd.DataFrame(report_rows)
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error generating injury report: {e}")
            return pd.DataFrame()