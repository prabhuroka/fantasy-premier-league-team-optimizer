
"""
View optimization results
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.prompt import Prompt
import questionary

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import (
    SAVED_TEAM_PATH, BASE_DIR, PREDICTIONS_DIR,
    FPL_RULES, POSITION_COLORS, POSITION_EMOJI
)


class ResultViewer:
    """View optimization results"""
    
    def __init__(self, console: Console):
        self.console = console
    
    def view_result(self, result_file: Path):
        """View optimization result"""
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            self.display_result(result, result_file.name)
            
        except Exception as e:
            self.console.print(f"[red]Error loading result: {e}[/red]")
    
    def display_result(self, result: Dict, filename: str):
        """Display optimization result"""
        self.console.clear()
        
        # Header
        gameweek = result.get('gameweek', 'N/A')
        self.console.print(f"\n[bold cyan]OPTIMIZATION RESULTS - GW{gameweek}[/bold cyan]")
        self.console.print(f"[dim]File: {filename}[/dim]")
        self.console.print("[dim]" + "─" * 60 + "[/dim]")
        
        # Current vs Optimized comparison
        if 'comparison' in result:
            comp = result['comparison']
            current = comp.get('current_team', {})
            optimized = comp.get('optimized_team', {})
            
            self.console.print("\n[bold cyan]TEAM COMPARISON[/bold cyan]")
            
            table = Table(box=box.SIMPLE)
            table.add_column("Metric", style="cyan", width=20)
            table.add_column("Current Team", style="yellow", width=15)
            table.add_column("Optimized Team", style="green", width=15)
            table.add_column("Change", width=15)
            
            # Points
            current_points = current.get('expected_points', 0)
            optimized_points = optimized.get('expected_points', 0)
            points_gain = optimized_points - current_points
            
            table.add_row(
                "Expected Points",
                f"{current_points:.1f}",
                f"{optimized_points:.1f}",
                f"[green]+{points_gain:.1f}[/green]" if points_gain > 0 else f"[red]{points_gain:.1f}[/red]"
            )
            
            # Cost
            current_cost = current.get('total_cost', 0)
            optimized_cost = optimized.get('total_cost', 0)
            cost_change = optimized_cost - current_cost
            
            table.add_row(
                "Total Cost",
                f"£{current_cost:.1f}M",
                f"£{optimized_cost:.1f}M",
                f"£{cost_change:+.1f}M"
            )
            
            # Bank Balance
            current_bank = current.get('bank_balance', 0)
            optimized_bank = optimized.get('bank_balance', 0)
            bank_change = optimized_bank - current_bank
            
            table.add_row(
                "Bank Balance",
                f"£{current_bank:.1f}M",
                f"£{optimized_bank:.1f}M",
                f"£{bank_change:+.1f}M"
            )
            
            # Team Validity
            current_valid = "✅" if current.get('is_valid', False) else "❌"
            optimized_valid = "✅" if optimized.get('is_valid', False) else "❌"
            
            table.add_row(
                "Team Valid",
                current_valid,
                optimized_valid,
                ""
            )
            
            self.console.print(table)
        
        # Show optimized team
        if 'optimized_team' in result:
            opt_team = result['optimized_team']
            self.display_optimized_team(opt_team)
        
        # Show transfers
        if 'transfer_plan' in result:
            transfers = result['transfer_plan']
            self.display_transfers(transfers)
        
        # Show chips
        if 'chip_analysis' in result:
            chips = result['chip_analysis']
            self.display_chips(chips)
        
        # Show captain
        if 'captain' in result:
            captain = result['captain']
            vice_captain = result.get('vice_captain', {})
            self.display_captain(captain, vice_captain)
        
        # Show recommendation
        if 'recommendation' in result:
            rec = result['recommendation']
            self.display_recommendation(rec)
        
        # Navigation
        self.console.print("\n[dim]" + "─" * 60 + "[/dim]")
        self.console.print("\n[cyan]Options:[/cyan]")
        self.console.print("  1. View Optimized Team Details")
        self.console.print("  2. View Starting XI")
        self.console.print("  3. View Bench")
        self.console.print("  4. ← Back")
        
        try:
            choice = questionary.select(
                "Select option:",
                choices=[
                    "View Optimized Team Details",
                    "View Starting XI", 
                    "View Bench",
                    "← Back"
                ]
            ).ask()
            
            if choice == "View Optimized Team Details":
                self.view_team_details(result)
            elif choice == "View Starting XI":
                self.view_starting_xi(result)
            elif choice == "View Bench":
                self.view_bench(result)
                
        except KeyboardInterrupt:
            pass
    
    def display_optimized_team(self, opt_team: Dict):
        """Display optimized team summary"""
        self.console.print("\n[bold cyan]OPTIMIZED TEAM[/bold cyan]")
        
        expected_points = opt_team.get('expected_points', 0)
        total_cost = opt_team.get('total_cost', 0)
        bank_balance = opt_team.get('bank_balance', 0)
        
        self.console.print(f"  Expected Points: [green]{expected_points:.1f}[/green]")
        self.console.print(f"  Total Cost: [yellow]£{total_cost:.1f}M[/yellow]")
        self.console.print(f"  Bank Balance: [cyan]£{bank_balance:.1f}M[/cyan]")
        
        if 'formation' in opt_team:
            formation = opt_team['formation']
            if isinstance(formation, dict):
                form_str = formation.get('formation', 'N/A')
                self.console.print(f"  Formation: {form_str}")
        
        if 'position_distribution' in opt_team:
            pos_dist = opt_team['position_distribution']
            pos_text = []
            for position, count in pos_dist.items():
                emoji = POSITION_EMOJI.get(position, '❓')
                color = POSITION_COLORS.get(position, 'white')
                pos_text.append(f"[{color}]{emoji} {position[:3]}: {count}[/{color}]")
            
            if pos_text:
                self.console.print("  Positions: " + " | ".join(pos_text))
    
    def display_transfers(self, transfers: Dict):
        """Display recommended transfers"""
        if transfers.get('num_transfers', 0) > 0:
            self.console.print("\n[bold cyan]RECOMMENDED TRANSFERS[/bold cyan]")
            
            self.console.print(f"  Transfers: {transfers['num_transfers']}")
            self.console.print(f"  Point Hit: {transfers['point_hit']}")
            self.console.print(f"  Expected Gain: [green]+{transfers['expected_gain']:.1f} points[/green]")
            self.console.print(f"  Net Gain: [green]+{transfers['net_gain']:.1f} points[/green]")
            
            if transfers.get('transfers'):
                self.console.print("\n  [dim]Transfer Details:[/dim]")
                for i, transfer in enumerate(transfers['transfers'], 1):
                    player_out = transfer.get('player_out', {})
                    player_in = transfer.get('player_in', {})
                    
                    self.console.print(
                        f"    {i}. [red]{player_out.get('name', 'Unknown')}[/red] → "
                        f"[green]{player_in.get('name', 'Unknown')}[/green]"
                    )
                    self.console.print(f"       Gain: +{transfer.get('net_gain', 0):.1f} points")
        else:
            self.console.print("\n[dim]No recommended transfers[/dim]")
    
    def display_chips(self, chips: Dict):
        """Display chip recommendations"""
        if chips.get('best_chip'):
            best_chip = chips['best_chip']
            self.console.print("\n[bold cyan]CHIP RECOMMENDATION[/bold cyan]")
            
            self.console.print(f"  Chip: [yellow]{best_chip['chip_name'].replace('_', ' ').title()}[/yellow]")
            self.console.print(f"  Gameweek: GW{best_chip['recommended_gw']}")
            self.console.print(f"  Expected Gain: [green]+{best_chip['expected_gain']:.1f} points[/green]")
            self.console.print(f"  Confidence: {best_chip['confidence']:.0%}")
            self.console.print(f"  Reason: {best_chip['reason']}")
    
    def display_captain(self, captain: Dict, vice_captain: Dict):
        """Display captain selection"""
        self.console.print("\n[bold cyan]CAPTAIN SELECTION[/bold cyan]")
        
        if captain.get('player_name'):
            self.console.print(f"  Captain: [yellow]© {captain['player_name']}[/yellow]")
            self.console.print(f"  Expected Points: {captain.get('captain_points', 0):.1f}")
            self.console.print(f"  Confidence: {captain.get('confidence', 0):.0%}")
        
        if vice_captain.get('player_name'):
            self.console.print(f"  Vice-Captain: [dim]VC {vice_captain['player_name']}[/dim]")
            self.console.print(f"  Expected Points: {vice_captain.get('predicted_points', 0):.1f}")
    
    def display_recommendation(self, recommendation: Dict):
        """Display final recommendation"""
        self.console.print("\n[bold cyan]FINAL RECOMMENDATION[/bold cyan]")
        
        action = recommendation.get('action', '').replace('_', ' ').title()
        reason = recommendation.get('reason', '')
        
        self.console.print(f"  Action: [yellow]{action}[/yellow]")
        self.console.print(f"  Reason: {reason}")
        
        if recommendation.get('expected_improvement', 0) > 0:
            self.console.print(f"  Expected Improvement: [green]+{recommendation['expected_improvement']:.1f} points[/green]")
        
        if recommendation.get('actions_needed'):
            self.console.print("\n  [dim]Actions Needed:[/dim]")
            for action in recommendation['actions_needed']:
                self.console.print(f"    • {action}")
    
    def view_team_details(self, result: Dict):
        """View detailed team information"""
        if 'optimized_team' not in result:
            return
        
        opt_team = result['optimized_team']
        players = opt_team.get('selected_players', [])
        
        if not players:
            return
        
        self.console.clear()
        self.console.print("\n[bold cyan]OPTIMIZED TEAM DETAILS[/bold cyan]")
        self.console.print("[dim]" + "─" * 60 + "[/dim]")
        
        # Group by position
        players_by_position = {}
        for player in players:
            if isinstance(player, dict):
                position = player.get('position', 'Unknown')
                if position not in players_by_position:
                    players_by_position[position] = []
                players_by_position[position].append(player)
        
        # Display by position
        for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            if position in players_by_position:
                self.console.print(f"\n[bold]{position}s[/bold]")
                
                for player in players_by_position[position]:
                    name = player.get('name', 'Unknown')
                    team = player.get('team_name', 'Unknown')
                    cost = player.get('cost', 0)
                    points = player.get('predicted_points', 0)
                    ppm = player.get('points_per_million', 0)
                    
                    emoji = POSITION_EMOJI.get(position, '❓')
                    color = POSITION_COLORS.get(position, 'white')
                    
                    self.console.print(
                        f"  [{color}]{emoji}[/{color}] {name} ({team}) - "
                        f"£{cost:.1f}M - {points:.1f} pts - {ppm:.1f} PPM"
                    )
        
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def view_starting_xi(self, result: Dict):
        """View starting XI"""
        if 'optimized_team' not in result or 'lineup' not in result:
            return
        
        opt_team = result['optimized_team']
        lineup = result['lineup']
        players = opt_team.get('selected_players', [])
        starting_xi_ids = lineup.get('starting_xi', [])
        
        if not players or not starting_xi_ids:
            return
        
        # Get starting XI players
        starting_xi = []
        for player in players:
            if isinstance(player, dict) and player.get('player_id') in starting_xi_ids:
                starting_xi.append(player)
        
        # Sort by predicted points
        starting_xi.sort(key=lambda x: x.get('predicted_points', 0), reverse=True)
        
        self.console.clear()
        self.console.print("\n[bold cyan]STARTING XI[/bold cyan]")
        self.console.print("[dim]" + "─" * 60 + "[/dim]")
        
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="dim", width=3)
        table.add_column("Player", width=20)
        table.add_column("Pos", width=5)
        table.add_column("Team", width=8)
        table.add_column("Cost", justify="right", width=8)
        table.add_column("Pts", justify="right", width=8)
        
        total_points = 0
        total_cost = 0
        
        for i, player in enumerate(starting_xi, 1):
            position = player.get('position', 'Unknown')
            emoji = POSITION_EMOJI.get(position, '❓')
            color = POSITION_COLORS.get(position, 'white')
            
            cost = player.get('cost', 0)
            points = player.get('predicted_points', 0)
            
            total_points += points
            total_cost += cost
            
            table.add_row(
                str(i),
                player.get('name', 'Unknown'),
                f"[{color}]{emoji}[/{color}]",
                player.get('team_name', 'Unknown'),
                f"£{cost:.1f}M",
                f"{points:.1f}"
            )
        
        self.console.print(table)
        
        # Totals
        self.console.print(f"\n  Total Points: [green]{total_points:.1f}[/green]")
        self.console.print(f"  Total Cost: [yellow]£{total_cost:.1f}M[/yellow]")
        
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def view_bench(self, result: Dict):
        """View bench players"""
        if 'optimized_team' not in result or 'lineup' not in result:
            return
        
        opt_team = result['optimized_team']
        lineup = result['lineup']
        players = opt_team.get('selected_players', [])
        bench_ids = lineup.get('bench', [])
        
        if not players or not bench_ids:
            return
        
        # Get bench players
        bench = []
        for player in players:
            if isinstance(player, dict) and player.get('player_id') in bench_ids:
                bench.append(player)
        
        if not bench:
            return
        
        self.console.clear()
        self.console.print("\n[bold cyan]BENCH[/bold cyan]")
        self.console.print("[dim]" + "─" * 60 + "[/dim]")
        
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="dim", width=3)
        table.add_column("Player", width=20)
        table.add_column("Pos", width=5)
        table.add_column("Team", width=8)
        table.add_column("Cost", justify="right", width=8)
        table.add_column("Pts", justify="right", width=8)
        
        total_points = 0
        total_cost = 0
        
        for i, player in enumerate(bench, 1):
            position = player.get('position', 'Unknown')
            emoji = POSITION_EMOJI.get(position, '❓')
            color = POSITION_COLORS.get(position, 'white')
            
            cost = player.get('cost', 0)
            points = player.get('predicted_points', 0)
            
            total_points += points
            total_cost += cost
            
            table.add_row(
                str(i),
                player.get('name', 'Unknown'),
                f"[{color}]{emoji}[/{color}]",
                player.get('team_name', 'Unknown'),
                f"£{cost:.1f}M",
                f"{points:.1f}"
            )
        
        self.console.print(table)
        
        # Totals
        self.console.print(f"\n  Bench Strength: [green]{total_points:.1f} points[/green]")
        self.console.print(f"  Bench Cost: [yellow]£{total_cost:.1f}M[/yellow]")
        
        # Show bench availability if available
        if lineup.get('bench_availability'):
            self.console.print(f"  Bench Availability: {lineup['bench_availability']:.0%}")
        
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")