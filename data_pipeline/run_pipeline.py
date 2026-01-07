"""
Main pipeline script for FPL Simulator
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.data_downloader import DataDownloader
from data_pipeline.database import FPLDatabase
from data_pipeline.data_processor import DataProcessor
from data_pipeline.config import CURRENT_SEASON


def run_pipeline():
    """Run the complete data pipeline"""
    print("\n" + "="*70)
    print("FPL SIMULATOR - DATA PIPELINE")
    print(f"Season: {CURRENT_SEASON}")
    print("="*70 + "\n")
    
    # Step 1: Download data
    print("STEP 1: Downloading data from GitHub...")
    downloader = DataDownloader()
    
    if not downloader.download_season_data():
        print("✗ Failed to download data. Exiting.")
        return False
    
    # Get available gameweeks
    available_gws = downloader.get_available_gameweeks()
    print(f"\n✓ Downloaded {len(available_gws)} gameweeks: {available_gws}")
    
    # Step 2: Initialize database
    print("\nSTEP 2: Initializing database...")
    db = FPLDatabase()
    
    if not db.connect():
        print("✗ Failed to connect to database. Exiting.")
        return False
    
    # Create tables (this will drop existing ones)
    print("Creating database tables...")
    if not db.create_tables():
        print("✗ Failed to create tables. Exiting.")
        db.close()
        return False
    
    # Step 3: Process data
    print("\nSTEP 3: Processing and loading data...")
    processor = DataProcessor(db)
    
    if not processor.process_all_data():
        print("✗ Failed to process data.")
        db.close()
        return False
    
    # Step 4: Show summary
    print("\nSTEP 4: Data summary...")
    
    # Get processed gameweeks
    processed_gws = processor.get_processed_gameweeks()
    print(f"✓ Processed {len(processed_gws)} gameweeks: {processed_gws}")
    
    # Get table counts
    counts = processor.get_data_summary()
    print("\nTable row counts:")
    for table, count in sorted(counts.items()):
        print(f"  {table:25}: {count:6} rows")
    
    # Show sample data
    print("\nSample data verification:")
    
    # Check players
    players_query = "SELECT COUNT(*) as count FROM players"
    players_result = db.execute_query(players_query)
    if players_result is not None:
        print(f"  Players: {players_result.iloc[0]['count']} records")
    
    # Check gameweek stats
    gw_stats_query = """
    SELECT 
        COUNT(*) as total,
        COUNT(DISTINCT gw) as gameweeks,
        COUNT(DISTINCT id) as players
    FROM player_gameweek_stats
    """
    gw_stats_result = db.execute_query(gw_stats_query)
    if gw_stats_result is not None:
        stats = gw_stats_result.iloc[0]
        print(f"  Gameweek stats: {stats['total']} records, {stats['gameweeks']} GWs, {stats['players']} players")
    
    # Sample query
    sample_query = """
    SELECT p.web_name, p.position, t.name as team_name, pgs.gw, pgs.total_points, pgs.minutes
    FROM player_gameweek_stats pgs
    JOIN players p ON pgs.id = p.player_id
    JOIN teams t ON p.team_code = t.code
    WHERE pgs.gw = 1 AND pgs.minutes > 0
    ORDER BY pgs.total_points DESC
    LIMIT 5
    """
    
    sample_result = db.execute_query(sample_query)
    if sample_result is not None and not sample_result.empty:
        print("\nTop performers in GW1:")
        for _, row in sample_result.iterrows():
            print(f"  {row['web_name']} ({row['position']}, {row['team_name']}): "
                  f"{row['total_points']} points, {row['minutes']} mins")
    
    # Close database
    db.close()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)