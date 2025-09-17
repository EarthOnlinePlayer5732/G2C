#!/usr/bin/env python3
"""
G2C Example: Advanced Console Game
A more sophisticated example using the G2C framework features.
"""

import sys
import os
import random

# Add src directory to path to import g2c
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from g2c import Console, GameState, InputHandler, print_header, get_yes_no, clear_screen
except ImportError:
    print("Error: Could not import G2C framework. Please ensure src/g2c.py exists.")
    sys.exit(1)

class AdvancedNumberGame:
    """Advanced number guessing game using G2C framework"""
    
    def __init__(self):
        self.state = GameState()
        self.state.set('score', 0)
        self.state.set('games_played', 0)
        self.state.set('total_attempts', 0)
    
    def setup_game(self):
        """Setup a new game"""
        print_header("Game Setup")
        
        # Get difficulty level
        print("Choose difficulty level:")
        print("1. Easy (1-50, 10 attempts)")
        print("2. Medium (1-100, 7 attempts)")
        print("3. Hard (1-200, 5 attempts)")
        
        difficulty = InputHandler.get_number("Enter choice (1-3): ", 1, 3)
        
        if difficulty == 1:
            self.max_number = 50
            self.max_attempts = 10
            self.difficulty_name = "Easy"
        elif difficulty == 2:
            self.max_number = 100
            self.max_attempts = 7
            self.difficulty_name = "Medium"
        else:
            self.max_number = 200
            self.max_attempts = 5
            self.difficulty_name = "Hard"
        
        self.secret_number = random.randint(1, self.max_number)
        self.attempts = 0
        
        clear_screen()
        print_header(f"Number Guessing Game - {self.difficulty_name} Mode")
        print(f"I'm thinking of a number between 1 and {self.max_number}.")
        print(f"You have {self.max_attempts} attempts to guess it!")
        print()
    
    def play_round(self):
        """Play a single round"""
        while self.attempts < self.max_attempts:
            try:
                guess = InputHandler.get_number(
                    f"Attempt {self.attempts + 1}/{self.max_attempts}: Enter your guess (1-{self.max_number}): ",
                    1, self.max_number
                )
                
                self.attempts += 1
                self.state.set('total_attempts', self.state.get('total_attempts') + 1)
                
                if guess == self.secret_number:
                    points = max(1, self.max_attempts - self.attempts + 1) * 10
                    if self.difficulty_name == "Medium":
                        points *= 2
                    elif self.difficulty_name == "Hard":
                        points *= 3
                    
                    self.state.set('score', self.state.get('score') + points)
                    
                    print(f"🎉 Congratulations! You guessed it in {self.attempts} attempts!")
                    print(f"💰 You earned {points} points!")
                    return True
                elif guess < self.secret_number:
                    print("📈 Too low! Try a higher number.")
                else:
                    print("📉 Too high! Try a lower number.")
                
            except KeyboardInterrupt:
                print("\n")
                if get_yes_no("Do you want to quit the game?"):
                    return False
                print()
        
        print(f"😞 Game over! The number was {self.secret_number}")
        return True
    
    def show_stats(self):
        """Show game statistics"""
        clear_screen()
        print_header("Game Statistics")
        
        games = self.state.get('games_played')
        score = self.state.get('score')
        attempts = self.state.get('total_attempts')
        
        print(f"Games Played: {games}")
        print(f"Total Score: {score}")
        print(f"Total Attempts: {attempts}")
        
        if games > 0:
            avg_attempts = attempts / games
            print(f"Average Attempts per Game: {avg_attempts:.1f}")
        
        print()
        input("Press Enter to continue...")
    
    def main_menu(self):
        """Display main menu"""
        while True:
            clear_screen()
            print_header("G2C Number Guessing Game")
            
            print("1. Play Game")
            print("2. View Statistics")
            print("3. About G2C")
            print("4. Quit")
            print()
            
            choice = InputHandler.get_choice("Enter your choice (1-4): ", ["1", "2", "3", "4"])
            
            if choice == "1":
                self.play_game()
            elif choice == "2":
                self.show_stats()
            elif choice == "3":
                self.show_about()
            elif choice == "4":
                clear_screen()
                print("Thanks for playing G2C Number Guessing Game!")
                break
    
    def play_game(self):
        """Play a complete game"""
        self.setup_game()
        
        if self.play_round():
            self.state.set('games_played', self.state.get('games_played') + 1)
        
        print()
        input("Press Enter to return to main menu...")
    
    def show_about(self):
        """Show information about G2C"""
        clear_screen()
        print_header("About G2C Framework")
        
        about_text = """
G2C (Game to Console) is a lightweight Python framework for developing
console-based games and applications. It provides utilities for:

• Console management (clearing, sizing, formatting)
• Game state management with checkpoints
• User input handling with validation
• Text formatting and display helpers

This number guessing game demonstrates the basic capabilities of the
G2C framework. The framework is designed to make console game development
easier and more enjoyable.

Visit: https://github.com/EarthOnlinePlayer5732/G2C
        """
        
        print(about_text.strip())
        print()
        input("Press Enter to continue...")

def main():
    """Main function"""
    try:
        game = AdvancedNumberGame()
        game.main_menu()
    except KeyboardInterrupt:
        clear_screen()
        print("\nGoodbye!")

if __name__ == "__main__":
    main()