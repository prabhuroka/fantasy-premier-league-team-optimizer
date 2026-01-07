"""
Run the feature engineering pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.database import FPLDatabase
from feature_engineering.feature_builder import FeatureBuilder
from feature_engineering.config import DATABASE_PATH, FEATURES_DIR
import pandas as pd


def main():
    """Run FIXED feature pipeline for ALL gameweeks"""
    print("\n" + "="*70)
    print("FPL SIMULATOR - FIXED FEATURE ENGINEERING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Connect to database
    print("STEP 1: Connecting to database...")
    db = FPLDatabase(DATABASE_PATH)
    
    if not db.connect():
        print("✗ Failed to connect to database. Exiting.")
        return False
    
    # Step 2: Initialize feature builder
    print("STEP 2: Initializing feature builder...")
    feature_builder = FeatureBuilder(db)
    
    # Step 3: Get latest gameweek
    print("\nSTEP 3: Getting latest gameweek...")
    query = "SELECT MAX(gw) as latest_gw FROM player_gameweek_stats WHERE season='2025-2026'"
    result = db.execute_query(query)
    
    if result is None or result.empty:
        print("✗ No data found")
        db.close()
        return False
    
    latest_gw = int(result.iloc[0]['latest_gw'])
    print(f"  Latest available gameweek: {latest_gw}")
    
    # Step 4: Generate training data
    print("\n" + "="*70)
    print(f"STEP 4: Generating FIXED training data (GW3-{latest_gw})...")
    print("="*70)

    all_training_data = []
    
    for target_gw in range(3, latest_gw + 1):
        print(f"\n📊 Processing GW{target_gw}...")
        
        # Get player history UP TO the previous gameweek
        print(f"  Getting player history up to GW{target_gw-1}...")
        
        history_query = """
            SELECT 
                pgs.id as player_id,
                pgs.gw as gameweek,
                pgs.season,
                pgs.minutes,
                pgs.total_points,
                pgs.goals_scored,
                pgs.assists,
                pgs.clean_sheets,
                pgs.goals_conceded,
                pgs.expected_goals,
                pgs.expected_assists,
                pgs.expected_goal_involvements,
                pgs.expected_goals_conceded,
                pgs.saves,
                pgs.bonus,
                pgs.selected_by_percent,
                pgs.now_cost,
                pgs.tackles,
                pgs.clearances_blocks_interceptions,
                pgs.influence,
                pgs.creativity,
                pgs.threat,
                pgs.ict_index,
                pgs.bps,
                pgs.yellow_cards,
                pgs.red_cards,
                pgs.own_goals,
                pgs.penalties_saved,
                pgs.penalties_missed,
                pgs.form
            FROM player_gameweek_stats pgs
            WHERE pgs.gw < ? 
              AND pgs.season = '2025-2026'
              AND pgs.minutes > 0
            ORDER BY pgs.id, pgs.gw
        """
        
        history_data = db.execute_query(history_query, (target_gw,))
        
        if history_data is None or history_data.empty:
            print(f"  ⚠ No history data for GW{target_gw}")
            continue
        
        print(f"  History records: {len(history_data)}")
        
        # Get player metadata
        print(f"  Getting player metadata...")
        
        metadata_query = """
            SELECT 
                player_id,
                COALESCE(position, 'UNK') as position,
                COALESCE(team_code, 0) as team_id,
                COALESCE(web_name, 'Unknown') as web_name
            FROM players
            WHERE season = '2025-2026'
        """
        
        player_data = db.execute_query(metadata_query)
        
        if player_data is None or player_data.empty:
            print(f"  ⚠ No player metadata found")
            continue
        
        # Create comprehensive features
        print(f"  Creating comprehensive features...")
        
        try:
            # Use the comprehensive feature creation method
            features_df = feature_builder.create_comprehensive_features(
                player_history=history_data,
                player_info=player_data,
                gameweek=target_gw - 1,  # Use previous GW for status features
                season="2025-2026"
            )
            
            if features_df.empty:
                print(f"  ⚠ No features generated")
                continue
            
            # Keep only the latest gameweek for each player
            if 'gameweek' in features_df.columns:
                features_df = features_df.sort_values('gameweek').groupby('player_id').tail(1)
            
            print(f"  Features created: {len(features_df)} players, {len(features_df.columns)} features")
            
            # Get target variable for this gameweek
            print(f"  Getting target variable for GW{target_gw}...")
            
            target_query = """
                SELECT id as player_id, total_points as actual_points
                FROM player_gameweek_stats 
                WHERE gw = ? 
                   AND season = '2025-2026'
            """
            
            target_data = db.execute_query(target_query, (target_gw,))
            
            if target_data is None or target_data.empty:
                print(f"  ⚠ No target data for GW{target_gw}")
                continue
            
            print(f"  Target records: {len(target_data)}")
            
            # Merge features with target
            print(f"  Merging features with target...")
            
            features_df['player_id'] = features_df['player_id'].astype(str)
            target_data['player_id'] = target_data['player_id'].astype(str)
            
            merged_df = features_df.merge(target_data, on='player_id', how='inner')
            
            if len(merged_df) > 0:
                merged_df['target_gameweek'] = target_gw
                all_training_data.append(merged_df)
                
                # Print summary
                print(f"  ✓ Training samples: {len(merged_df)}")
                
                # Check critical features
                critical_features = ['status_numeric', 'playing_probability', 'form_adjusted', 
                                   'saves_per_90', 'conceded_per_90', 'def_actions_per_90',
                                   'xa_per_90', 'xg_per_90', 'rolling_points_3']
                
                present = [f for f in critical_features if f in merged_df.columns]
                print(f"  ✓ Critical features present: {len(present)}/{len(critical_features)}")
                
                # Check if we have goalkeeper data
                if 'position' in merged_df.columns:
                    gk_count = len(merged_df[merged_df['position'] == 'Goalkeeper'])
                    if gk_count > 0:
                        print(f"  ✓ Goalkeepers in data: {gk_count}")
            else:
                print(f"  ⚠ No matching players found")
                
        except Exception as e:
            print(f"  ✗ Error processing GW{target_gw}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all training data
    if all_training_data:
        training_df = pd.concat(all_training_data, ignore_index=True)
        
        # Final cleanup
        print(f"\nPerforming final cleanup on {len(training_df)} samples...")
        
        # Remove any remaining cost/value columns
        cost_cols = [col for col in training_df.columns if any(keyword in col.lower() 
                    for keyword in ['cost', 'value', 'price', 'million', 'transfer'])]
        if cost_cols:
            print(f"  Removing {len(cost_cols)} cost/value columns")
            training_df = training_df.drop(columns=cost_cols)
        
        # Save training data
        fixed_file = os.path.join(FEATURES_DIR, 'fixed_training_data.csv')
        training_df.to_csv(fixed_file, index=False)
        
        # Also save as complete_training_data.csv for backward compatibility
        complete_file = os.path.join(FEATURES_DIR, 'complete_training_data.csv')
        training_df.to_csv(complete_file, index=False)
        
        print(f"\n✓ Training data saved to: {fixed_file}")
        print(f"  Also saved as: {complete_file}")
        print(f"  Total samples: {len(training_df)}")
        print(f"  Total features: {len(training_df.columns)}")
        
        # Show feature categories
        position_specific = [col for col in training_df.columns if 'per_90' in col or 'prob' in col]
        rolling_features = [col for col in training_df.columns if 'rolling_' in col]
        status_features = [col for col in training_df.columns if any(keyword in col 
                          for keyword in ['status', 'probability', 'chance'])]
        
        print(f"\nFeature Categories:")
        print(f"  Position-specific: {len(position_specific)} features")
        print(f"  Rolling features: {len(rolling_features)} features")
        print(f"  Status/availability: {len(status_features)} features")
        
        if 'actual_points' in training_df.columns:
            print(f"\nTarget statistics:")
            print(f"  Min: {training_df['actual_points'].min()}")
            print(f"  Max: {training_df['actual_points'].max()}")
            print(f"  Mean: {training_df['actual_points'].mean():.2f}")
            print(f"  Std: {training_df['actual_points'].std():.2f}")
    else:
        print("✗ No training data generated")
        db.close()
        return False
    
    # Close database
    db.close()
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETED WITH FIXES!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)