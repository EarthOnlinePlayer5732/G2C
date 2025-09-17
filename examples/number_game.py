#!/usr/bin/env python3
"""
G2C Example: Simple Console Game
A basic number guessing game to demonstrate G2C framework concepts.
"""

import random
import sys

class ConsoleGame:
    """Base class for console games"""
    
    def __init__(self, title="G2C Game"):
        self.title = title
        self.running = False
    
    def display_header(self):
        """Display game header"""
        print("=" * 50)
        print(f" {self.title}")
        print("=" * 50)
    
    def start(self):
        """Start the game"""
        self.running = True
        self.display_header()
        self.game_loop()
    
    def game_loop(self):
        """Main game loop - to be implemented by subclasses"""
        pass
    
    def quit_game(self):
        """Quit the game"""
        self.running = False
        print("\nThanks for playing!")

class NumberGuessingGame(ConsoleGame):
    """A simple number guessing game"""
    
    def __init__(self):
        super().__init__("Number Guessing Game - G2C Example")
        self.secret_number = random.randint(1, 100)
        self.attempts = 0
        self.max_attempts = 7
    
    def game_loop(self):
        """Number guessing game loop"""
        print(f"I'm thinking of a number between 1 and 100.")
        print(f"You have {self.max_attempts} attempts to guess it!")
        print()
        
        while self.running and self.attempts < self.max_attempts:
            try:
                guess = int(input(f"Attempt {self.attempts + 1}/{self.max_attempts}: Enter your guess: "))
                self.attempts += 1
                
                if guess == self.secret_number:
                    print(f"🎉 Congratulations! You guessed it in {self.attempts} attempts!")
                    self.running = False
                elif guess < self.secret_number:
                    print("📈 Too low! Try a higher number.")
                else:
                    print("📉 Too high! Try a lower number.")
                
                if self.attempts >= self.max_attempts and self.running:
                    print(f"😞 Game over! The number was {self.secret_number}")
                    self.running = False
                    
            except ValueError:
                print("❌ Please enter a valid number!")
            except KeyboardInterrupt:
                print("\n")
                self.quit_game()
                break

def main():
    """Main function"""
    print("Welcome to G2C - Game to Console Framework!")
    print("This is a simple example of what can be built with G2C.\n")
    
    game = NumberGuessingGame()
    game.start()

if __name__ == "__main__":
    main()