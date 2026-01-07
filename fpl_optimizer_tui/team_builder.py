"""
Team building interface with real-time validation
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
import questionary
import pandas as pd

from fpl_optimizer_tui.config import (
    SAVED_TEAM_PATH, BASE_DIR, PREDICTIONS_DIR,
    FPL_RULES, POSITION_COLORS, POSITION_EMOJI
)


class TeamBuilder:
    """Build and validate FPL teams"""
    
    def __init__(self, console: Console):
        self.console = console
        self.players_cache = None
        
    def load_players(self) -> List[Dict]:
        """Load players from latest predictions"""
        if self.players_cache is not None:
            return self.players_cache
        
        # Find latest predictions file
        prediction_files = list(PREDICTIONS_DIR.glob("raw_points_predictions_gw*.csv"))
        
        if not prediction_files:
            self.console.print("[yellow]No prediction files found[/yellow]")
            self.console.print("Please run the ML pipeline first.")
            return []
        
        # Get latest file
        latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = pd.read_csv(latest_file)
            players = []
            
            for _, row in df.iterrows():
                players.append({
                    'player_id': int(row.get('player_id', 0)),
                    'name': row.get('web_name', f"Player {row.get('player_id', 'Unknown')}"),
                    'position': row.get('position', 'Unknown'),
                    'team_id': int(row.get('team_id', 0)),
                    'team_name': row.get('team_name', f"Team {row.get('team_id', 'Unknown')}"),
                    'cost': float(row.get('now_cost', 0)) / 10,  # Convert to £M
                    'predicted_points': float(row.get('predicted_points', 0)),
                    'points_per_million': float(row.get('points_per_million', 0))
                })
            
            self.players_cache = players
            return players
            
        except Exception as e:
            self.console.print(f"[red]Error loading players: {e}[/red]")
            return []
    
    def get_player_by_id(self, player_id: int) -> Optional[Dict]:
        """Get player by ID"""
        players = self.load_players()
        for player in players:
            if player['player_id'] == player_id:
                return player
        return None
    
    def get_player_by_name(self, name: str) -> Optional[Dict]:
        """Get player by name (partial match)"""
        players = self.load_players()
        name_lower = name.lower()
        for player in players:
            if name_lower in player['name'].lower():
                return player
        return None
    
    def validate_team(self, team_players: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate team against FPL rules"""
        errors = []
        
        # Check squad size
        if len(team_players) > FPL_RULES['squad_size']:
            errors.append(f"Too many players: {len(team_players)}/{FPL_RULES['squad_size']}")
        elif len(team_players) < FPL_RULES['squad_size']:
            errors.append(f"Incomplete team: {len(team_players)}/{FPL_RULES['squad_size']}")
        
        # Check budget
        total_cost = sum(p['cost'] for p in team_players)
        if total_cost > FPL_RULES['budget']:
            errors.append(f"Over budget: £{total_cost:.1f}M > £{FPL_RULES['budget']:.1f}M")
        
        # Check position limits
        position_counts = {}
        for position in FPL_RULES['position_limits'].keys():
            position_counts[position] = 0
        
        for player in team_players:
            position = player['position']
            if position in position_counts:
                position_counts[position] += 1
        
        for position, (min_limit, max_limit) in FPL_RULES['position_limits'].items():
            count = position_counts.get(position, 0)
            if count < min_limit:
                errors.append(f"Need at least {min_limit} {position.lower()}(s), have {count}")
            elif count > max_limit:
                errors.append(f"Too many {position.lower()}(s): {count} > {max_limit}")
        
        # Check team limits
        team_counts = {}
        for player in team_players:
            team_id = player['team_id']
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        for team_id, count in team_counts.items():
            if count > FPL_RULES['team_limit']:
                team_name = team_players[0]['team_name'] if team_players else f"Team {team_id}"
                errors.append(f"Too many players from {team_name}: {count} > {FPL_RULES['team_limit']}")
        
        # Check unique players
        player_ids = [p['player_id'] for p in team_players]
        if len(player_ids) != len(set(player_ids)):
            errors.append("Duplicate players in team")
        
        return len(errors) == 0, errors
    
    def calculate_team_stats(self, team_players: List[Dict]) -> Dict:
        """Calculate team statistics"""
        total_cost = sum(p['cost'] for p in team_players)
        total_points = sum(p['predicted_points'] for p in team_players)
        bank_balance = FPL_RULES['budget'] - total_cost
        
        position_counts = {}
        for player in team_players:
            position = player['position']
            position_counts[position] = position_counts.get(position, 0) + 1
        
        return {
            'total_cost': total_cost,
            'bank_balance': bank_balance,
            'total_points': total_points,
            'position_counts': position_counts,
            'player_count': len(team_players)
        }
    
    def display_team(self, team_players: List[Dict], title: str = "Your Team"):
        """Display team in a table"""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        self.console.print("[dim]" + "─" * 60 + "[/dim]")
        
        if not team_players:
            self.console.print("[yellow]No players in team[/yellow]")
            return
        
        # Calculate stats
        stats = self.calculate_team_stats(team_players)
        
        # Display stats
        self.console.print(f"👥 Players: {stats['player_count']}/{FPL_RULES['squad_size']}")
        self.console.print(f"💰 Cost: £{stats['total_cost']:.1f}M | Bank: £{stats['bank_balance']:.1f}M")
        self.console.print(f"⭐ Expected Points: {stats['total_points']:.1f}")
        
        # Display position counts
        position_text = []
        for position, count in stats['position_counts'].items():
            emoji = POSITION_EMOJI.get(position, '❓')
            color = POSITION_COLORS.get(position, 'white')
            min_limit, max_limit = FPL_RULES['position_limits'].get(position, (0, 0))
            position_text.append(f"[{color}]{emoji} {position[:3]}: {count}/{max_limit}[/{color}]")
        
        if position_text:
            self.console.print("📊 " + " | ".join(position_text))
        
        # Display players table
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Player", width=20)
        table.add_column("Pos", width=5)
        table.add_column("Team", width=8)
        table.add_column("Cost", justify="right", width=8)
        table.add_column("Pts", justify="right", width=8)
        
        for i, player in enumerate(team_players, 1):
            position = player['position']
            emoji = POSITION_EMOJI.get(position, '❓')
            color = POSITION_COLORS.get(position, 'white')
            
            table.add_row(
                str(i),
                player['name'],
                f"[{color}]{emoji}[/{color}]",
                player['team_name'],
                f"£{player['cost']:.1f}M",
                f"{player['predicted_points']:.1f}"
            )
        
        self.console.print(table)
        
        # Validate and show errors
        is_valid, errors = self.validate_team(team_players)
        if is_valid:
            self.console.print("[green]✓ Team is valid[/green]")
        else:
            self.console.print("[red]✗ Team has issues:[/red]")
            for error in errors:
                self.console.print(f"  • {error}")
    
    def create_new_team(self):
        """Create a new team from scratch"""
        self.console.print("\n[cyan]Creating new team...[/cyan]")
        
        # Load players
        players = self.load_players()
        if not players:
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return
        
        team_players = []
        
        while True:
            self.console.clear()
            self.display_team(team_players, "Team Builder")
            
            # Show available positions
            stats = self.calculate_team_stats(team_players)
            position_counts = stats.get('position_counts', {})
            
            self.console.print("\n[bold cyan]Available Slots:[/bold cyan]")
            for position, (min_limit, max_limit) in FPL_RULES['position_limits'].items():
                current = position_counts.get(position, 0)
                remaining = max_limit - current
                emoji = POSITION_EMOJI.get(position, '❓')
                color = POSITION_COLORS.get(position, 'white')
                
                status = f"[{color}]{remaining}[/{color}]" if remaining > 0 else "[red]0[/red]"
                self.console.print(f"  {emoji} {position}: {current}/{max_limit} (Remaining: {status})")
            
            # Show team slots
            remaining_slots = FPL_RULES['squad_size'] - len(team_players)
            self.console.print(f"\n📋 Team Slots: {len(team_players)}/{FPL_RULES['squad_size']} (Remaining: {remaining_slots})")
            
            # Menu
            self.console.print("\n[cyan]Options:[/cyan]")
            self.console.print("  1. Add Player")
            self.console.print("  2. Remove Player")
            self.console.print("  3. Save Team")
            self.console.print("  4. Cancel")
            
            try:
                choice = IntPrompt.ask(
                    "\n[cyan]Select option[/cyan]",
                    choices=["1", "2", "3", "4"],
                    show_choices=False
                )
                
                if choice == 1:
                    self.add_player_interactive(team_players, players)
                elif choice == 2:
                    self.remove_player_interactive(team_players)
                elif choice == 3:
                    if self.save_team(team_players):
                        return
                elif choice == 4:
                    if Confirm.ask("[yellow]Cancel without saving?[/yellow]"):
                        return
                        
            except (KeyboardInterrupt, ValueError):
                return
    
    def add_player_interactive(self, team_players: List[Dict], all_players: List[Dict]):
        """Add a player to the team"""
        # Group players by position
        players_by_position = {}
        for player in all_players:
            position = player['position']
            if position not in players_by_position:
                players_by_position[position] = []
            players_by_position[position].append(player)
        
        # Get available positions (not at max limit)
        stats = self.calculate_team_stats(team_players)
        position_counts = stats.get('position_counts', {})
        
        available_positions = []
        for position, (min_limit, max_limit) in FPL_RULES['position_limits'].items():
            current = position_counts.get(position, 0)
            if current < max_limit:
                available_positions.append(position)
        
        if not available_positions:
            self.console.print("[red]All positions at maximum![/red]")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return
        
        # Select position
        self.console.print("\n[cyan]Select position:[/cyan]")
        for i, position in enumerate(available_positions, 1):
            emoji = POSITION_EMOJI.get(position, '❓')
            color = POSITION_COLORS.get(position, 'white')
            current = position_counts.get(position, 0)
            min_limit, max_limit = FPL_RULES['position_limits'][position]
            self.console.print(f"  {i}. [{color}]{emoji} {position}[/{color}] ({current}/{max_limit})")
        
        try:
            pos_choice = IntPrompt.ask(
                "\n[cyan]Position[/cyan]",
                choices=[str(i) for i in range(1, len(available_positions) + 1)],
                show_choices=False
            )
            
            position = available_positions[pos_choice - 1]
            position_players = players_by_position.get(position, [])
            
            if not position_players:
                self.console.print(f"[yellow]No {position}s available[/yellow]")
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
                return
            
            # Sort by predicted points
            position_players.sort(key=lambda x: x['predicted_points'], reverse=True)
            
            # Show top players
            self.console.print(f"\n[cyan]Top {position}s:[/cyan]")
            for i, player in enumerate(position_players[:10], 1):
                emoji = POSITION_EMOJI.get(position, '❓')
                color = POSITION_COLORS.get(position, 'white')
                
                # Check if already in team
                in_team = any(p['player_id'] == player['player_id'] for p in team_players)
                team_marker = "✓ " if in_team else ""
                
                self.console.print(
                    f"  {i}. {team_marker}[{color}]{emoji}[/{color}] {player['name']} "
                    f"({player['team_name']}) - £{player['cost']:.1f}M - {player['predicted_points']:.1f} pts"
                )
            
            player_choice = IntPrompt.ask(
                f"\n[cyan]Select {position} to add[/cyan]",
                choices=[str(i) for i in range(1, min(11, len(position_players) + 1))],
                show_choices=False
            )
            
            selected_player = position_players[player_choice - 1]
            
            # Check if already in team
            if any(p['player_id'] == selected_player['player_id'] for p in team_players):
                self.console.print("[yellow]Player already in team![/yellow]")
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
                return
            
            # Check budget
            stats = self.calculate_team_stats(team_players)
            remaining_budget = stats['bank_balance']
            
            if selected_player['cost'] > remaining_budget:
                self.console.print(f"[red]Cannot afford! Need £{selected_player['cost']:.1f}M, "
                                  f"but only £{remaining_budget:.1f}M remaining[/red]")
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
                return
            
            # Check team limit
            team_id = selected_player['team_id']
            team_count = sum(1 for p in team_players if p['team_id'] == team_id)
            
            if team_count >= FPL_RULES['team_limit']:
                self.console.print(f"[red]Already have {team_count} players from {selected_player['team_name']}. "
                                  f"Limit is {FPL_RULES['team_limit']}[/red]")
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
                return
            
            # Add player
            team_players.append(selected_player)
            self.console.print(f"[green]✓ Added {selected_player['name']} to team[/green]")
            
            if len(team_players) >= FPL_RULES['squad_size']:
                self.console.print("[yellow]Team is full![/yellow]")
            
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            
        except (KeyboardInterrupt, ValueError):
            return
    
    def remove_player_interactive(self, team_players: List[Dict]):
        """Remove a player from the team"""
        if not team_players:
            self.console.print("[yellow]No players to remove![/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return
        
        self.console.print("\n[cyan]Select player to remove:[/cyan]")
        
        for i, player in enumerate(team_players, 1):
            position = player['position']
            emoji = POSITION_EMOJI.get(position, '❓')
            color = POSITION_COLORS.get(position, 'white')
            
            self.console.print(
                f"  {i}. [{color}]{emoji}[/{color}] {player['name']} "
                f"({player['team_name']}) - £{player['cost']:.1f}M"
            )
        
        try:
            choice = IntPrompt.ask(
                "\n[cyan]Player number[/cyan]",
                choices=[str(i) for i in range(1, len(team_players) + 1)],
                show_choices=False
            )
            
            if 1 <= choice <= len(team_players):
                removed_player = team_players.pop(choice - 1)
                self.console.print(f"[green]✓ Removed {removed_player['name']}[/green]")
            else:
                self.console.print("[red]Invalid selection[/red]")
            
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            
        except (KeyboardInterrupt, ValueError):
            return
    
    def edit_team(self):
        """Edit existing team"""
        # Load current team
        current_team = self.load_saved_team()
        if not current_team:
            self.console.print("[yellow]No team to edit[/yellow]")
            return
        
        # Convert player IDs to player objects
        team_players = []
        for player_id in current_team['players']:
            player = self.get_player_by_id(player_id)
            if player:
                team_players.append(player)
            else:
                self.console.print(f"[yellow]Player {player_id} not found in predictions[/yellow]")
        
        # Load all players
        all_players = self.load_players()
        
        # Edit team
        while True:
            self.console.clear()
            self.display_team(team_players, "Edit Team")
            
            self.console.print("\n[cyan]Options:[/cyan]")
            self.console.print("  1. Add Player")
            self.console.print("  2. Remove Player")
            self.console.print("  3. Save Changes")
            self.console.print("  4. Cancel")
            
            try:
                choice = IntPrompt.ask(
                    "\n[cyan]Select option[/cyan]",
                    choices=["1", "2", "3", "4"],
                    show_choices=False
                )
                
                if choice == 1:
                    self.add_player_interactive(team_players, all_players)
                elif choice == 2:
                    self.remove_player_interactive(team_players)
                elif choice == 3:
                    if self.save_team(team_players):
                        return
                elif choice == 4:
                    if Confirm.ask("[yellow]Cancel without saving?[/yellow]"):
                        return
                        
            except (KeyboardInterrupt, ValueError):
                return
    
    def view_current_team(self):
        """View current saved team"""
        current_team = self.load_saved_team()
        if not current_team:
            self.console.print("[yellow]No team saved[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return
        
        # Convert player IDs to player objects
        team_players = []
        for player_id in current_team['players']:
            player = self.get_player_by_id(player_id)
            if player:
                team_players.append(player)
            else:
                self.console.print(f"[yellow]Player {player_id} not found in predictions[/yellow]")
        
        self.display_team(team_players, "Current Saved Team")
        
        # Show team info
        self.console.print(f"\n💰 Bank Balance: £{current_team['bank_balance']:.1f}M")
        self.console.print(f"🔄 Free Transfers: {current_team['free_transfers']}")
        
        if current_team.get('chips_available'):
            self.console.print(f"🎲 Available Chips: {', '.join(current_team['chips_available'])}")
        
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def save_team(self, team_players: List[Dict]) -> bool:
        """Save team to file"""
        # Validate team
        is_valid, errors = self.validate_team(team_players)
        
        if not is_valid:
            self.console.print("\n[red]⚠ Cannot save invalid team:[/red]")
            for error in errors:
                self.console.print(f"  • {error}")
            
            if not Confirm.ask("\n[yellow]Save anyway?[/yellow]"):
                return False
        
        # Calculate bank balance
        total_cost = sum(p['cost'] for p in team_players)
        bank_balance = FPL_RULES['budget'] - total_cost
        
        # Create team data
        team_data = {
            'players': [p['player_id'] for p in team_players],
            'bank_balance': bank_balance,
            'free_transfers': 1,
            'chips_available': [
                'wildcard',
                'free_hit', 
                'triple_captain',
                'bench_boost'
            ],
            'total_points': 0,
            'overall_rank': 0
        }
        
        # Save to file
        try:
            with open(SAVED_TEAM_PATH, 'w') as f:
                json.dump(team_data, f, indent=2)
            
            self.console.print(f"\n[green]✓ Team saved to {SAVED_TEAM_PATH.name}[/green]")
            self.console.print(f"💰 Bank Balance: £{bank_balance:.1f}M")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error saving team: {e}[/red]")
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            return False
    
    def load_saved_team(self) -> Optional[Dict]:
        """Load saved team from file"""
        if not SAVED_TEAM_PATH.exists():
            return None
        
        try:
            with open(SAVED_TEAM_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.console.print(f"[red]Error loading team: {e}[/red]")
            return None