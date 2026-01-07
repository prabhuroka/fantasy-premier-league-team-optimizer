"""
Simplified database operations for FPL Simulator
"""
import sqlite3
import pandas as pd
import os
from typing import Optional, Dict, List
from datetime import datetime

from data_pipeline.config import DATABASE_PATH, SQLITE_PRAGMAS, CURRENT_SEASON


class FPLDatabase:
    """Simplified SQLite database operations"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database connection"""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Set pragmas for better performance
            for pragma, value in SQLITE_PRAGMAS.items():
                self.conn.execute(f"PRAGMA {pragma} = {value}")
            
            print(f"✓ Connected to database: {self.db_path}")
            return True
            
        except sqlite3.Error as e:
            print(f"✗ Database connection error: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_tables(self) -> bool:
        """Create simplified tables matching GitHub data structure"""
        try:
            cursor = self.conn.cursor()
            
            # Drop existing tables
            tables_to_drop = [
                'players', 'teams', 'player_stats', 'player_gameweek_stats',
                'fixtures', 'matches', 'player_match_stats'
            ]
            
            for table in tables_to_drop:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                except:
                    pass
            
            # Create players table (from players.csv)
            cursor.execute('''
            CREATE TABLE players (
                player_id INTEGER PRIMARY KEY,
                player_code INTEGER,
                first_name TEXT,
                second_name TEXT,
                web_name TEXT,
                team_code INTEGER,
                position TEXT,
                season TEXT DEFAULT '2025-2026',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create teams table (from teams.csv)
            cursor.execute('''
            CREATE TABLE teams (
                team_id INTEGER PRIMARY KEY,
                code INTEGER,
                name TEXT,
                short_name TEXT,
                strength INTEGER,
                strength_overall_home INTEGER,
                strength_overall_away INTEGER,
                strength_attack_home INTEGER,
                strength_attack_away INTEGER,
                strength_defence_home INTEGER,
                strength_defence_away INTEGER,
                pulse_id INTEGER,
                elo REAL,
                fotmob_name TEXT,
                season TEXT DEFAULT '2025-2026',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create player_stats table (from playerstats.csv)
            cursor.execute('''
            CREATE TABLE player_stats (
                id INTEGER,
                status TEXT,
                chance_of_playing_next_round REAL,
                chance_of_playing_this_round REAL,
                now_cost REAL,
                now_cost_rank INTEGER,
                now_cost_rank_type INTEGER,
                cost_change_event INTEGER,
                cost_change_event_fall INTEGER,
                cost_change_start INTEGER,
                cost_change_start_fall INTEGER,
                selected_by_percent REAL,
                selected_rank INTEGER,
                selected_rank_type INTEGER,
                total_points INTEGER,
                event_points INTEGER,
                points_per_game REAL,
                points_per_game_rank INTEGER,
                points_per_game_rank_type INTEGER,
                bonus INTEGER,
                bps INTEGER,
                form REAL,
                form_rank INTEGER,
                form_rank_type INTEGER,
                value_form REAL,
                value_season REAL,
                dreamteam_count INTEGER,
                transfers_in INTEGER,
                transfers_in_event INTEGER,
                transfers_out INTEGER,
                transfers_out_event INTEGER,
                ep_next REAL,
                ep_this REAL,
                expected_goals REAL,
                expected_assists REAL,
                expected_goal_involvements REAL,
                expected_goals_conceded REAL,
                expected_goals_per_90 REAL,
                expected_assists_per_90 REAL,
                expected_goal_involvements_per_90 REAL,
                expected_goals_conceded_per_90 REAL,
                influence REAL,
                influence_rank INTEGER,
                influence_rank_type INTEGER,
                creativity REAL,
                creativity_rank INTEGER,
                creativity_rank_type INTEGER,
                threat REAL,
                threat_rank INTEGER,
                threat_rank_type INTEGER,
                ict_index REAL,
                ict_index_rank INTEGER,
                ict_index_rank_type INTEGER,
                corners_and_indirect_freekicks_order INTEGER,
                direct_freekicks_order INTEGER,
                penalties_order INTEGER,
                gw INTEGER,
                set_piece_threat REAL,
                first_name TEXT,
                second_name TEXT,
                web_name TEXT,
                news TEXT,
                news_added TEXT,
                minutes INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                goals_conceded INTEGER,
                own_goals INTEGER,
                penalties_saved INTEGER,
                penalties_missed INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                saves INTEGER,
                starts INTEGER,
                defensive_contribution INTEGER,
                corners_and_indirect_freekicks_text TEXT,
                direct_freekicks_text TEXT,
                penalties_text TEXT,
                saves_per_90 REAL,
                clean_sheets_per_90 REAL,
                goals_conceded_per_90 REAL,
                starts_per_90 REAL,
                defensive_contribution_per_90 REAL,
                tackles INTEGER,
                clearances_blocks_interceptions INTEGER,
                recoveries INTEGER,
                season TEXT DEFAULT '2025-2026',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, gw, season)
            )
            ''')
            
            # Create player_gameweek_stats table (from player_gameweek_stats.csv)
            cursor.execute('''
            CREATE TABLE player_gameweek_stats (
                id INTEGER,
                first_name TEXT,
                second_name TEXT,
                web_name TEXT,
                status TEXT,
                news TEXT,
                news_added TEXT,
                now_cost REAL,
                now_cost_rank INTEGER,
                now_cost_rank_type INTEGER,
                selected_by_percent REAL,
                selected_rank INTEGER,
                selected_rank_type INTEGER,
                form REAL,
                form_rank INTEGER,
                form_rank_type INTEGER,
                event_points INTEGER,
                cost_change_event INTEGER,
                cost_change_event_fall INTEGER,
                cost_change_start INTEGER,
                cost_change_start_fall INTEGER,
                transfers_in_event INTEGER,
                transfers_out_event INTEGER,
                value_form REAL,
                value_season REAL,
                ep_next REAL,
                ep_this REAL,
                points_per_game REAL,
                points_per_game_rank INTEGER,
                points_per_game_rank_type INTEGER,
                chance_of_playing_next_round REAL,
                chance_of_playing_this_round REAL,
                influence_rank INTEGER,
                influence_rank_type INTEGER,
                creativity_rank INTEGER,
                creativity_rank_type INTEGER,
                threat_rank INTEGER,
                threat_rank_type INTEGER,
                ict_index_rank INTEGER,
                ict_index_rank_type INTEGER,
                corners_and_indirect_freekicks_order INTEGER,
                direct_freekicks_order INTEGER,
                penalties_order INTEGER,
                set_piece_threat REAL,
                corners_and_indirect_freekicks_text TEXT,
                direct_freekicks_text TEXT,
                penalties_text TEXT,
                expected_goals_per_90 REAL,
                expected_assists_per_90 REAL,
                expected_goal_involvements_per_90 REAL,
                expected_goals_conceded_per_90 REAL,
                saves_per_90 REAL,
                clean_sheets_per_90 REAL,
                goals_conceded_per_90 REAL,
                starts_per_90 REAL,
                defensive_contribution_per_90 REAL,
                gw INTEGER,
                total_points INTEGER,
                minutes INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                goals_conceded INTEGER,
                own_goals INTEGER,
                penalties_saved INTEGER,
                penalties_missed INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                saves INTEGER,
                starts INTEGER,
                bonus INTEGER,
                bps INTEGER,
                transfers_in INTEGER,
                transfers_out INTEGER,
                dreamteam_count INTEGER,
                expected_goals REAL,
                expected_assists REAL,
                expected_goal_involvements REAL,
                expected_goals_conceded REAL,
                influence REAL,
                creativity REAL,
                threat REAL,
                ict_index REAL,
                tackles INTEGER,
                clearances_blocks_interceptions INTEGER,
                recoveries INTEGER,
                defensive_contribution INTEGER,
                season TEXT DEFAULT '2025-2026',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, gw, season)
            )
            ''')
            
            # Create matches table (from matches.csv)
            cursor.execute('''
            CREATE TABLE matches (
                gameweek REAL,
                kickoff_time TEXT,
                home_team REAL,
                home_team_elo REAL,
                home_score REAL,
                away_score REAL,
                away_team REAL,
                away_team_elo REAL,
                finished INTEGER,
                match_id TEXT PRIMARY KEY,
                match_url TEXT,
                home_possession REAL,
                away_possession REAL,
                home_expected_goals_xg REAL,
                away_expected_goals_xg REAL,
                home_total_shots REAL,
                away_total_shots REAL,
                home_shots_on_target REAL,
                away_shots_on_target REAL,
                home_big_chances REAL,
                away_big_chances REAL,
                home_big_chances_missed REAL,
                away_big_chances_missed REAL,
                home_accurate_passes REAL,
                away_accurate_passes REAL,
                home_accurate_passes_pct REAL,
                away_accurate_passes_pct REAL,
                home_fouls_committed REAL,
                away_fouls_committed REAL,
                home_corners REAL,
                away_corners REAL,
                home_xg_open_play REAL,
                away_xg_open_play REAL,
                home_xg_set_play REAL,
                away_xg_set_play REAL,
                home_non_penalty_xg REAL,
                away_non_penalty_xg REAL,
                home_xg_on_target_xgot REAL,
                away_xg_on_target_xgot REAL,
                home_shots_off_target REAL,
                away_shots_off_target REAL,
                home_blocked_shots REAL,
                away_blocked_shots REAL,
                home_hit_woodwork REAL,
                away_hit_woodwork REAL,
                home_shots_inside_box REAL,
                away_shots_inside_box REAL,
                home_shots_outside_box REAL,
                away_shots_outside_box REAL,
                home_passes REAL,
                away_passes REAL,
                home_own_half REAL,
                away_own_half REAL,
                home_opposition_half REAL,
                away_opposition_half REAL,
                home_accurate_long_balls REAL,
                away_accurate_long_balls REAL,
                home_accurate_long_balls_pct REAL,
                away_accurate_long_balls_pct REAL,
                home_accurate_crosses REAL,
                away_accurate_crosses REAL,
                home_accurate_crosses_pct REAL,
                away_accurate_crosses_pct REAL,
                home_throws REAL,
                away_throws REAL,
                home_touches_in_opposition_box REAL,
                away_touches_in_opposition_box REAL,
                home_offsides REAL,
                away_offsides REAL,
                home_yellow_cards REAL,
                away_yellow_cards REAL,
                home_red_cards REAL,
                away_red_cards REAL,
                home_tackles_won REAL,
                away_tackles_won REAL,
                home_tackles_won_pct REAL,
                away_tackles_won_pct REAL,
                home_interceptions REAL,
                away_interceptions REAL,
                home_blocks REAL,
                away_blocks REAL,
                home_clearances REAL,
                away_clearances REAL,
                home_keeper_saves REAL,
                away_keeper_saves REAL,
                home_duels_won REAL,
                away_duels_won REAL,
                home_ground_duels_won REAL,
                away_ground_duels_won REAL,
                home_ground_duels_won_pct REAL,
                away_ground_duels_won_pct REAL,
                home_aerial_duels_won REAL,
                away_aerial_duels_won REAL,
                home_aerial_duels_won_pct REAL,
                away_aerial_duels_won_pct REAL,
                home_successful_dribbles REAL,
                away_successful_dribbles REAL,
                home_successful_dribbles_pct REAL,
                away_successful_dribbles_pct REAL,
                fotmob_id REAL,
                stats_processed INTEGER,
                player_stats_processed INTEGER,
                home_distance_covered REAL,
                away_distance_covered REAL,
                home_walking_distance REAL,
                away_walking_distance REAL,
                home_running_distance REAL,
                away_running_distance REAL,
                home_sprinting_distance REAL,
                away_sprinting_distance REAL,
                home_number_of_sprints REAL,
                away_number_of_sprints REAL,
                home_top_speed REAL,
                away_top_speed REAL,
                tournament TEXT,
                season TEXT DEFAULT '2025-2026',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create player_match_stats table (from playermatchstats.csv)
            cursor.execute('''
            CREATE TABLE player_match_stats (
                player_id INTEGER,
                gw INTEGER,
                match_id TEXT,
                minutes_played INTEGER,
                goals INTEGER,
                assists INTEGER,
                total_shots INTEGER,
                xg REAL,
                xa REAL,
                shots_on_target INTEGER,
                successful_dribbles INTEGER,
                big_chances_missed INTEGER,
                touches_opposition_box INTEGER,
                touches INTEGER,
                accurate_passes INTEGER,
                accurate_passes_percent REAL,
                chances_created INTEGER,
                final_third_passes INTEGER,
                accurate_crosses INTEGER,
                accurate_crosses_percent REAL,
                accurate_long_balls INTEGER,
                accurate_long_balls_percent REAL,
                tackles_won INTEGER,
                interceptions INTEGER,
                recoveries INTEGER,
                blocks INTEGER,
                clearances INTEGER,
                headed_clearances INTEGER,
                dribbled_past INTEGER,
                duels_won INTEGER,
                duels_lost INTEGER,
                ground_duels_won INTEGER,
                ground_duels_won_percent REAL,
                aerial_duels_won INTEGER,
                aerial_duels_won_percent REAL,
                was_fouled INTEGER,
                fouls_committed INTEGER,
                saves INTEGER,
                goals_conceded INTEGER,
                xgot_faced REAL,
                goals_prevented REAL,
                sweeper_actions INTEGER,
                gk_accurate_passes INTEGER,
                gk_accurate_long_balls INTEGER,
                dispossessed INTEGER,
                high_claim INTEGER,
                corners INTEGER,
                saves_inside_box INTEGER,
                offsides INTEGER,
                successful_dribbles_percent REAL,
                tackles_won_percent REAL,
                xgot REAL,
                tackles INTEGER,
                start_min INTEGER,
                finish_min INTEGER,
                team_goals_conceded INTEGER,
                penalties_scored INTEGER,
                penalties_missed INTEGER,
                top_speed REAL,
                distance_covered REAL,
                walking_distance REAL,
                running_distance REAL,
                sprinting_distance REAL,
                number_of_sprints INTEGER,
                defensive_contributions INTEGER,
                season TEXT DEFAULT '2025-2026',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, match_id, season)
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX idx_player_gw ON player_gameweek_stats(id, gw, season)')
            cursor.execute('CREATE INDEX idx_player_stats_gw ON player_stats(id, gw, season)')
            cursor.execute('CREATE INDEX idx_matches_gw ON matches(gameweek, season)')
            cursor.execute('CREATE INDEX idx_player_match_stats ON player_match_stats(player_id, match_id)')
            
            self.conn.commit()
            print("✓ Database tables created successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error creating tables: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def execute_query(self, query: str, params: tuple = ()) -> Optional[pd.DataFrame]:
        """Execute a SQL query and return results as DataFrame"""
        try:
            if not self.conn:
                self.connect()
            
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
            
        except Exception as e:
            print(f"✗ Query execution error: {e}")
            return None
    
    def execute_write(self, query: str, params: tuple = ()) -> bool:
        """Execute a write SQL query"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"✗ Write execution error: {e}")
            return False
    
    def save_dataframe(self, table_name: str, df: pd.DataFrame, 
                      if_exists: str = 'append') -> bool:
        """Save a DataFrame to the database"""
        try:
            if not self.conn:
                self.connect()
            
            df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
            print(f"✓ Saved {len(df)} rows to {table_name}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving to {table_name}: {e}")
            return False
    
    def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables"""
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        counts = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            counts[table_name] = count
        
        return counts
    
    def get_latest_gameweek(self, season: str = "2025-2026") -> int:
        """
        Get the latest available gameweek from database
        
        Returns:
            Latest gameweek number (e.g., 17)
        """
        query = """
            SELECT MAX(gw) as latest_gw 
            FROM player_gameweek_stats 
            WHERE season = ?
        """
        result = self.execute_query(query, (season,))
        if result is not None and not result.empty:
            return int(result.iloc[0]['latest_gw'])
        return 1  # Default to GW1 if no data