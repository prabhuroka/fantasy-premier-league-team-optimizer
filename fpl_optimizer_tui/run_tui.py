"""
Run the FPL Optimizer TUI
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Check dependencies
try:
    from rich.console import Console
    import questionary
    from rich.prompt import Prompt
except ImportError:
    print("Missing required packages. Installing...")
    
    # Try to install
    try:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "rich", "questionary"
        ])
        print("Packages installed successfully!")
    except:
        print("Failed to install packages. Please install manually:")
        print("  pip install rich questionary")
        sys.exit(1)

# Run the TUI
try:
    from main import main
    main()
except ImportError as e:
    print(f"Error: {e}")
    print("\nMake sure you're in the correct directory and all files are present.")
    sys.exit(1)