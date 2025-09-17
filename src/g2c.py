"""
G2C - Game to Console Framework
A simple framework for building console-based games and applications.
"""

import os
import sys
from typing import List, Dict, Any

__version__ = "0.1.0"

class Console:
    """Console utilities for game development"""
    
    @staticmethod
    def clear():
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def get_size():
        """Get console size (width, height)"""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except OSError:
            return 80, 24  # Default fallback
    
    @staticmethod
    def center_text(text: str, width: int = None) -> str:
        """Center text in the console"""
        if width is None:
            width, _ = Console.get_size()
        return text.center(width)
    
    @staticmethod
    def print_box(text: str, char: str = "=", padding: int = 2):
        """Print text in a box"""
        width, _ = Console.get_size()
        content_width = width - (2 * padding) - 2
        
        # Top border
        print(char * width)
        
        # Content
        lines = text.split('\n')
        for line in lines:
            if len(line) > content_width:
                # Wrap long lines
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) <= content_width:
                        current_line += word + " "
                    else:
                        print(f"{char}{current_line.ljust(content_width + 2 * padding)}{char}")
                        current_line = word + " "
                if current_line:
                    print(f"{char}{current_line.ljust(content_width + 2 * padding)}{char}")
            else:
                print(f"{char}{line.center(content_width + 2 * padding)}{char}")
        
        # Bottom border
        print(char * width)

class GameState:
    """Simple game state management"""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def set(self, key: str, value: Any):
        """Set a value in the game state"""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the game state"""
        return self.data.get(key, default)
    
    def save_checkpoint(self):
        """Save current state as a checkpoint"""
        self.history.append(self.data.copy())
    
    def restore_checkpoint(self, index: int = -1):
        """Restore a previous checkpoint"""
        if self.history:
            self.data = self.history[index].copy()

class InputHandler:
    """Handle various types of user input"""
    
    @staticmethod
    def get_choice(prompt: str, choices: List[str], case_sensitive: bool = False) -> str:
        """Get user choice from a list of options"""
        if not case_sensitive:
            choices = [choice.lower() for choice in choices]
        
        while True:
            user_input = input(prompt)
            if not case_sensitive:
                user_input = user_input.lower()
            
            if user_input in choices:
                return user_input
            else:
                print(f"Invalid choice. Please choose from: {', '.join(choices)}")
    
    @staticmethod
    def get_number(prompt: str, min_val: int = None, max_val: int = None) -> int:
        """Get a number from user with optional range validation"""
        while True:
            try:
                value = int(input(prompt))
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue
                return value
            except ValueError:
                print("Please enter a valid number.")

# Convenience functions
def clear_screen():
    """Clear the console screen"""
    Console.clear()

def print_header(title: str):
    """Print a formatted header"""
    Console.print_box(title, "=")

def get_yes_no(prompt: str) -> bool:
    """Get a yes/no response from user"""
    choice = InputHandler.get_choice(f"{prompt} (y/n): ", ["y", "yes", "n", "no"])
    return choice in ["y", "yes"]