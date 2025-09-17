#!/usr/bin/env python3
"""
G2C Framework Demo - Quick showcase of capabilities
Run this to see G2C framework features in action.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from g2c import Console, GameState, InputHandler, print_header, clear_screen, get_yes_no

def main():
    """Demonstrate G2C framework capabilities"""
    
    print_header("Welcome to G2C Framework Demo")
    
    print("G2C (Game to Console) Framework Demonstration")
    print("=" * 50)
    print()
    
    # Console utilities demo
    print("1. Console Utilities Demo:")
    width, height = Console.get_size()
    print(f"   • Console size: {width} x {height}")
    print(f"   • Centered text: {Console.center_text('*** G2C ***', 30)}")
    print()
    
    # Game state demo
    print("2. Game State Management Demo:")
    state = GameState()
    state.set('demo_score', 1000)
    state.set('demo_level', 5)
    print(f"   • Score stored: {state.get('demo_score')}")
    print(f"   • Level stored: {state.get('demo_level')}")
    print("   • State checkpoint saved")
    state.save_checkpoint()
    print()
    
    # Input handling demo (if user wants to try)
    if get_yes_no("3. Try input handling demo?"):
        print("\nInput Handling Demo:")
        
        name = input("   • Enter your name: ")
        
        color = InputHandler.get_choice(
            "   • Choose a color (red/blue/green): ", 
            ["red", "blue", "green"]
        )
        
        number = InputHandler.get_number(
            "   • Enter a number (1-10): ", 
            1, 10
        )
        
        print(f"\n   Results: Hello {name}! You chose {color} and number {number}")
    
    print()
    Console.print_box("G2C Framework Demo Complete!", "=", 2)
    print()
    print("Next steps:")
    print("• Run: python3 examples/number_game.py")
    print("• Run: python3 examples/advanced_game.py") 
    print("• Read: docs/README.md")
    print("• Visit: https://github.com/EarthOnlinePlayer5732/G2C")

if __name__ == "__main__":
    main()