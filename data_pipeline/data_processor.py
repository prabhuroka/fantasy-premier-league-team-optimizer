"""
Process and load data from CSV files to database
"""
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from data_pipeline.config import RAW_DATA_DIR, CURRENT_SEASON
from data_pipeline.database import FPLDatabase


class DataProcessor:
    """Process CSV files and load into database"""
    
    def __init__(self, db: FPLDatabase):
        self.db = db
        self.season = CURRENT_SEASON
        self.season_path = os.path.join(RAW_DATA_DIR, self.season)
    
    def process_all_data(self) -> bool:
        """Process all data for the season"""
        print(f"\n{'='*60}")
        print(f"Processing {self.season} data")
        print(f"{'='*60}")
        
        try:
            # Process master files
            print("\n1. Processing master files...")
            master_success = self._process_master_files()
            
            # Process gameweek data
            print("\n2. Processing gameweek data...")
            gw_success = self._process_gameweek_data()
            
            return master_success and gw_success
            
        except Exception as e:
            print(f"✗ Error processing data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_master_files(self) -> bool:
        """Process master CSV files"""
        master_files = {
            'players': 'players.csv',
            'teams': 'teams.csv',
            'player_stats': 'playerstats.csv',
        }
        
        all_success = True
        
        for table_name, file_name in master_files.items():
            file_path = os.path.join(self.season_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"  ✗ {file_name} not found")
                all_success = False
                continue
            
            try:
                print(f"  Processing {file_name}...")
                
                # Read CSV
                df = pd.read_csv(file_path)
                print(f"    Read {len(df)} rows")
                
                # Add season column
                df['season'] = self.season
                df['last_updated'] = datetime.now()
                
                # Save to database
                if self.db.save_dataframe(table_name, df, if_exists='replace'):
                    print(f"    ✓ Saved to {table_name} table")
                else:
                    print(f"    ✗ Failed to save to {table_name}")
                    all_success = False
                    
            except Exception as e:
                print(f"    ✗ Error processing {file_name}: {e}")
                all_success = False
        
        return all_success
    
    def _process_gameweek_data(self) -> bool:
        """Process all gameweek data"""
        by_gameweek_path = os.path.join(self.season_path, 'By Gameweek')
        
        if not os.path.exists(by_gameweek_path):
            print(f"  ✗ 'By Gameweek' folder not found")
            return False
        
        # Get all gameweek folders
        gw_folders = []
        for item in os.listdir(by_gameweek_path):
            if item.startswith('GW') and os.path.isdir(os.path.join(by_gameweek_path, item)):
                try:
                    gw_num = int(item[2:])
                    gw_folders.append((gw_num, item))
                except ValueError:
                    continue
        
        if not gw_folders:
            print(f"  ✗ No gameweek folders found")
            return False
        
        # Sort by gameweek number
        gw_folders.sort()
        print(f"  Found {len(gw_folders)} gameweek folders")
        
        # Process each gameweek
        all_success = True
        
        for gw_num, gw_folder in gw_folders:
            print(f"\n  Processing {gw_folder}...")
            gw_success = self._process_single_gameweek(gw_num, gw_folder)
            
            if not gw_success:
                all_success = False
                print(f"    ✗ Failed to process {gw_folder}")
        
        return all_success
    
    def _process_single_gameweek(self, gw_num: int, gw_folder: str) -> bool:
        """Process data for a single gameweek"""
        gw_path = os.path.join(self.season_path, 'By Gameweek', gw_folder)
        
        # Files to process for this gameweek
        gw_files = {
            'matches': 'matches.csv',
            'player_gameweek_stats': 'player_gameweek_stats.csv',
            'player_stats': 'playerstats.csv',
            'player_match_stats': 'playermatchstats.csv',
        }
        
        all_success = True
        
        for table_name, file_name in gw_files.items():
            file_path = os.path.join(gw_path, file_name)
            
            if not os.path.exists(file_path):
                # Some files might be optional
                if table_name == 'player_match_stats':
                    print(f"    ⚠ {file_name} not found (optional)")
                    continue
                else:
                    print(f"    ✗ {file_name} not found")
                    all_success = False
                    continue
            
            try:
                # Read CSV
                df = pd.read_csv(file_path)
                
                # Add gameweek and season columns
                if 'gw' not in df.columns and table_name != 'matches':
                    df['gw'] = gw_num
                
                df['season'] = self.season
                df['last_updated'] = datetime.now()
                
                # For matches table, ensure gameweek column exists
                if table_name == 'matches' and 'gameweek' not in df.columns:
                    df['gameweek'] = gw_num
                
                print(f"    {file_name}: {len(df)} rows")
                
                # Save to database
                if self.db.save_dataframe(table_name, df, if_exists='append'):
                    print(f"      ✓ Saved to {table_name}")
                else:
                    print(f"      ✗ Failed to save to {table_name}")
                    all_success = False
                    
            except Exception as e:
                print(f"    ✗ Error processing {file_name}: {e}")
                all_success = False
        
        return all_success
    
    def get_processed_gameweeks(self) -> List[int]:
        """Get list of gameweeks that have been processed"""
        query = """
        SELECT DISTINCT gw 
        FROM player_gameweek_stats 
        WHERE season = ?
        ORDER BY gw
        """
        
        result = self.db.execute_query(query, (self.season,))
        if result is not None and not result.empty:
            return result['gw'].tolist()
        return []
    
    def get_data_summary(self) -> Dict[str, int]:
        """Get summary of loaded data"""
        counts = self.db.get_table_counts()
        return counts