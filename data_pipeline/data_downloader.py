"""
Simplified data downloader for FPL-Elo-Insights GitHub repo
"""
import os
import shutil
import requests
import zipfile
import subprocess
from datetime import datetime
from typing import Optional, Tuple

from data_pipeline.config import RAW_DATA_DIR, CURRENT_SEASON, GITHUB_REPO_URL


class DataDownloader:
    """Download data from GitHub repository"""
    
    def __init__(self):
        self.season = CURRENT_SEASON
        self.season_path = os.path.join(RAW_DATA_DIR, self.season)
        os.makedirs(self.season_path, exist_ok=True)
    
    def download_season_data(self) -> bool:
        """
        Download entire season data from GitHub
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Downloading {self.season} data from GitHub")
        print(f"{'='*60}")
        
        # Remove existing data if it exists
        if os.path.exists(self.season_path):
            print(f"Removing existing data...")
            shutil.rmtree(self.season_path)
        
        # Try different download methods
        methods = [
            self._download_via_git,
            self._download_via_zip,
        ]
        
        for method in methods:
            print(f"\nTrying {method.__name__}...")
            if method():
                print(f"✓ Successfully downloaded {self.season} data")
                return True
        
        print(f"✗ All download methods failed")
        return False
    
    def _download_via_git(self) -> bool:
        """Download using git clone"""
        try:
            # Create temp directory
            temp_dir = f"{self.season_path}_temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Clone repository
            print(f"  Cloning repository...")
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', '--branch', 'main',
                 GITHUB_REPO_URL, temp_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  ✗ Git clone failed: {result.stderr[:200]}")
                return False
            
            # Find season data
            source_path = os.path.join(temp_dir, 'data', self.season)
            if not os.path.exists(source_path):
                # Try alternative location
                source_path = os.path.join(temp_dir, self.season)
                if not os.path.exists(source_path):
                    print(f"  ✗ Could not find {self.season} data in repo")
                    shutil.rmtree(temp_dir)
                    return False
            
            # Move data to season path
            print(f"  Moving data to {self.season_path}...")
            shutil.move(source_path, self.season_path)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            # Verify download
            return self._verify_download()
            
        except Exception as e:
            print(f"  ✗ Git download error: {e}")
            return False
    
    def _download_via_zip(self) -> bool:
        """Download as ZIP archive"""
        try:
            zip_url = f"{GITHUB_REPO_URL}/archive/refs/heads/main.zip"
            print(f"  Downloading ZIP from {zip_url}...")
            
            response = requests.get(zip_url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Save zip file
            zip_path = os.path.join(RAW_DATA_DIR, 'temp.zip')
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract zip
            print(f"  Extracting ZIP...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            
            # Find and move season data
            extracted_path = os.path.join(RAW_DATA_DIR, 'FPL-Elo-Insights-main')
            source_path = os.path.join(extracted_path, 'data', self.season)
            
            if not os.path.exists(source_path):
                source_path = os.path.join(extracted_path, self.season)
            
            if os.path.exists(source_path):
                # Move to season path
                if os.path.exists(self.season_path):
                    shutil.rmtree(self.season_path)
                shutil.move(source_path, self.season_path)
            
            # Clean up
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(extracted_path):
                shutil.rmtree(extracted_path, ignore_errors=True)
            
            # Verify download
            return self._verify_download()
            
        except Exception as e:
            print(f"  ✗ ZIP download error: {e}")
            return False
    
    def _verify_download(self) -> bool:
        """Verify that data was downloaded successfully"""
        required_files = [
            'players.csv',
            'teams.csv',
            'playerstats.csv',
        ]
        
        # Check for By Gameweek folder
        by_gameweek_path = os.path.join(self.season_path, 'By Gameweek')
        if not os.path.exists(by_gameweek_path):
            print(f"  ✗ 'By Gameweek' folder not found")
            return False
        
        # Check for at least one gameweek folder
        gw_folders = [d for d in os.listdir(by_gameweek_path) 
                     if d.startswith('GW') and os.path.isdir(os.path.join(by_gameweek_path, d))]
        
        if not gw_folders:
            print(f"  ✗ No gameweek folders found")
            return False
        
        print(f"  ✓ Found {len(gw_folders)} gameweek folders")
        
        # Check for required files in first gameweek
        sample_gw = gw_folders[0]
        sample_gw_path = os.path.join(by_gameweek_path, sample_gw)
        
        for file in required_files:
            file_path = os.path.join(sample_gw_path, file)
            if not os.path.exists(file_path):
                print(f"  ✗ Missing {file} in {sample_gw}")
                return False
        
        print(f"  ✓ All required files found in {sample_gw}")
        return True
    
    def get_available_gameweeks(self) -> list:
        """Get list of available gameweeks"""
        by_gameweek_path = os.path.join(self.season_path, 'By Gameweek')
        if not os.path.exists(by_gameweek_path):
            return []
        
        gw_folders = []
        for item in os.listdir(by_gameweek_path):
            if item.startswith('GW') and os.path.isdir(os.path.join(by_gameweek_path, item)):
                try:
                    gw_num = int(item[2:])
                    gw_folders.append((gw_num, item))
                except ValueError:
                    continue
        
        # Sort by gameweek number
        gw_folders.sort()
        return [gw for _, gw in gw_folders]