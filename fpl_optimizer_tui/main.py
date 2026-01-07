"""
FPL Optimizer - Simple Terminal UI
"""
import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich import box
import questionary

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Now import local modules
from config import (
    BASE_DIR, OPTIMIZATION_DIR, SAVED_TEAM_PATH,
    OPTIMIZATION_CMD
)
from team_builder import TeamBuilder
from result_viewer import ResultViewer


class FPLOptimizerTUI:
    """Simple Terminal UI for FPL Optimizer"""
    
    def __init__(self):
        self.console = Console()
        self.team_builder = TeamBuilder(self.console)
        self.result_viewer = ResultViewer(self.console)
        self.is_running = True
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str = "FPL OPTIMIZER"):
        """Print application header"""
        self.clear_screen()
        
        header = f"""
┌{'─' * 60}┐
│ {'🎯 ' + title:^58} │
└{'─' * 60}┘
        """
        self.console.print(header)
    
    def run(self):
        """Main application loop"""
        while self.is_running:
            self.show_main_menu()
    
    def show_main_menu(self):
        """Display main menu"""
        self.print_header("FPL OPTIMIZER")
        
        self.console.print("\n[bold cyan]Main Menu[/bold cyan]\n")
        
        choices = [
            ("🏗️  Build/Edit Team", self.build_team),
            ("🚀 Optimize Current Team", self.optimize_team),
            ("📊 View Optimization Results", self.view_results),
            ("ℹ️  View Team Rules", self.show_rules),
            ("🚪 Exit", self.exit_app)
        ]
        
        for i, (text, _) in enumerate(choices, 1):
            self.console.print(f"  {i}. {text}")
        
        try:
            choice = IntPrompt.ask(
                "\n[cyan]Select option[/cyan]",
                choices=[str(i) for i in range(1, len(choices) + 1)],
                show_choices=False
            )
            
            if 1 <= choice <= len(choices):
                _, action = choices[choice - 1]
                action()
                
        except KeyboardInterrupt:
            self.exit_app()
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def build_team(self):
        """Build or edit team"""
        self.print_header("TEAM BUILDER")
        
        # Check if team exists
        if SAVED_TEAM_PATH.exists():
            self.console.print("[green]✓ Found existing team[/green]")
            
            choices = [
                "Edit existing team",
                "Create new team",
                "View current team",
                "← Back"
            ]
            
            selection = questionary.select(
                "What would you like to do?",
                choices=choices
            ).ask()
            
            if not selection or selection == "← Back":
                return
            elif selection == "Edit existing team":
                self.team_builder.edit_team()
            elif selection == "Create new team":
                if Confirm.ask("[yellow]Create new team? This will overwrite current team.[/yellow]"):
                    self.team_builder.create_new_team()
            elif selection == "View current team":
                self.team_builder.view_current_team()
        else:
            self.console.print("[yellow]No team found. Let's create one![/yellow]")
            if Confirm.ask("\n[cyan]Create new team?[/cyan]"):
                self.team_builder.create_new_team()
    
    def optimize_team(self):
        """Run optimization on current team"""
        self.print_header("OPTIMIZE TEAM")
        
        # Check if team exists
        if not SAVED_TEAM_PATH.exists():
            self.console.print("[red]✗ No team found![/red]")
            self.console.print("Please build a team first.")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return
        
        # Check for predictions
        predictions_dir = BASE_DIR / "data" / "predictions"
        prediction_files = list(predictions_dir.glob("raw_points_predictions_gw*.csv"))
        
        if not prediction_files:
            self.console.print("[yellow]⚠ No prediction files found[/yellow]")
            self.console.print("You need to run the ML pipeline first.")
            self.console.print("\nRun: python ml_model/run_raw_point_pipeline.py")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return
        
        # Get latest gameweek
        latest_gw = 1
        for file in prediction_files:
            import re
            match = re.search(r'gw(\d+)', file.name.lower())
            if match:
                gw = int(match.group(1))
                latest_gw = max(latest_gw, gw)
        
        self.console.print(f"\n[cyan]Latest predictions: GW{latest_gw}[/cyan]")
        
        # Ask which gameweek to optimize for
        try:
            gameweek = IntPrompt.ask(
                "[cyan]Optimize for gameweek[/cyan]",
                default=latest_gw,
                show_default=True
            )
            
            if gameweek < 1:
                self.console.print("[red]Gameweek must be positive[/red]")
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
                return
            
        except (KeyboardInterrupt, ValueError):
            return
        
        # Run optimization
        self.console.print(f"\n[yellow]Running optimization for GW{gameweek}...[/yellow]")
        self.console.print("[dim]This may take a minute...[/dim]\n")
        
        try:
            # Build command with specific gameweek
            cmd = OPTIMIZATION_CMD.copy()
            if gameweek != latest_gw:
                cmd.extend(["--gameweek", str(gameweek)])
            
            # Run optimization
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=BASE_DIR
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ Optimization completed successfully![/green]")
                
                # Find the result file
                result_pattern = OPTIMIZATION_DIR / f"complete_optimization_gw{gameweek}.json"
                if result_pattern.exists():
                    self.console.print(f"\n[cyan]Results saved to: {result_pattern.name}[/cyan]")
                    
                    if Confirm.ask("\n[yellow]View results now?[/yellow]"):
                        self.view_single_result(gameweek)
                else:
                    # Try to find any recent result file
                    result_files = list(OPTIMIZATION_DIR.glob("complete_optimization_gw*.json"))
                    if result_files:
                        latest_result = max(result_files, key=os.path.getctime)
                        self.console.print(f"\n[cyan]Results saved to: {latest_result.name}[/cyan]")
                        
                        if Confirm.ask("\n[yellow]View latest results?[/yellow]"):
                            self.result_viewer.view_result(latest_result)
                    else:
                        self.console.print("[yellow]⚠ Could not find result file[/yellow]")
            else:
                self.console.print("[red]✗ Optimization failed![/red]")
                if result.stderr:
                    self.console.print(f"[dim]{result.stderr[:200]}...[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Error running optimization: {e}[/red]")
        
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def view_results(self):
        """View optimization results"""
        self.print_header("VIEW RESULTS")
        
        # Find all result files
        result_files = list(OPTIMIZATION_DIR.glob("complete_optimization_gw*.json"))
        
        if not result_files:
            self.console.print("[yellow]No optimization results found[/yellow]")
            self.console.print("Run optimization first to generate results.")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return
        
        # Sort by gameweek (extract number from filename)
        def extract_gw(filename):
            import re
            match = re.search(r'gw(\d+)', filename.name.lower())
            return int(match.group(1)) if match else 0
        
        result_files.sort(key=extract_gw, reverse=True)
        
        self.console.print(f"[green]Found {len(result_files)} optimization results[/green]\n")
        
        choices = []
        for file in result_files:
            gw = extract_gw(file)
            size_mb = file.stat().st_size / (1024 * 1024)
            choices.append(f"GW{gw:02d} - {size_mb:.1f} MB")
        
        choices.append("← Back")
        
        selection = questionary.select(
            "Select result to view:",
            choices=choices
        ).ask()
        
        if not selection or selection == "← Back":
            return
        
        # Extract gameweek from selection
        try:
            gw_str = selection.split(" - ")[0]
            gameweek = int(gw_str.replace("GW", ""))
            
            # Find the file
            result_file = None
            for file in result_files:
                if extract_gw(file) == gameweek:
                    result_file = file
                    break
            
            if result_file:
                self.result_viewer.view_result(result_file)
            else:
                self.console.print("[red]Result file not found[/red]")
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
                
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/red]")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def view_single_result(self, gameweek: int):
        """View result for a specific gameweek"""
        result_file = OPTIMIZATION_DIR / f"complete_optimization_gw{gameweek}.json"
        
        if result_file.exists():
            self.result_viewer.view_result(result_file)
        else:
            # Try pattern matching
            result_files = list(OPTIMIZATION_DIR.glob(f"*gw{gameweek}*.json"))
            if result_files:
                self.result_viewer.view_result(result_files[0])
            else:
                self.console.print(f"[yellow]No results found for GW{gameweek}[/yellow]")
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def show_rules(self):
        """Show FPL rules"""
        self.print_header("FPL RULES")
        
        from fpl_optimizer_tui.config import FPL_RULES
        
        self.console.print("\n[bold cyan]FPL Team Rules[/bold cyan]\n")
        
        self.console.print(f"💰 [yellow]Budget:[/yellow] £{FPL_RULES['budget']:.1f}M")
        self.console.print(f"👥 [yellow]Squad Size:[/yellow] {FPL_RULES['squad_size']} players")
        self.console.print(f"⚽ [yellow]Starting XI:[/yellow] {FPL_RULES['starting_xi']} players")
        self.console.print(f"📋 [yellow]Bench:[/yellow] {FPL_RULES['bench_size']} players\n")
        
        self.console.print("[bold cyan]Position Limits:[/bold cyan]")
        for position, (min_limit, max_limit) in FPL_RULES['position_limits'].items():
            emoji = {
                'Goalkeeper': '🧤',
                'Defender': '🛡️',
                'Midfielder': '⚽',
                'Forward': '🎯'
            }.get(position, '❓')
            self.console.print(f"  {emoji} {position}: {min_limit}-{max_limit}")
        
        self.console.print(f"\n🏆 [yellow]Team Limit:[/yellow] Max {FPL_RULES['team_limit']} players from one team")
        
        self.console.print("\n[bold cyan]Starting XI Minimum:[/bold cyan]")
        for position, min_count in FPL_RULES['formation_min'].items():
            self.console.print(f"  {position}: At least {min_count}")
        
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def exit_app(self):
        """Exit the application"""
        if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
            self.is_running = False
            self.console.print("\n[cyan]Good luck with your FPL team! 🍀[/cyan]\n")


def main():
    """Main entry point"""
    app = FPLOptimizerTUI()
    
    try:
        app.run()
    except KeyboardInterrupt:
        app.console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        app.console.print(f"\n[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()