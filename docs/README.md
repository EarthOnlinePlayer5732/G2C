# G2C Framework Documentation

## Overview

G2C (Game to Console) is a lightweight Python framework designed to simplify the development of console-based games and applications. It provides a set of utilities and classes that handle common tasks in console application development.

## Core Components

### Console Class

The `Console` class provides utilities for console management:

- `clear()` - Clear the console screen
- `get_size()` - Get console dimensions (width, height)
- `center_text(text, width)` - Center text within a given width
- `print_box(text, char, padding)` - Print text in a decorative box

### GameState Class

The `GameState` class handles game state management:

- `set(key, value)` - Store a value in the game state
- `get(key, default)` - Retrieve a value from the game state
- `save_checkpoint()` - Save current state as a checkpoint
- `restore_checkpoint(index)` - Restore a previous checkpoint

### InputHandler Class

The `InputHandler` class provides robust user input handling:

- `get_choice(prompt, choices, case_sensitive)` - Get user choice from options
- `get_number(prompt, min_val, max_val)` - Get numeric input with validation

## Convenience Functions

- `clear_screen()` - Shorthand for Console.clear()
- `print_header(title)` - Print a formatted header
- `get_yes_no(prompt)` - Get a yes/no response from user

## Quick Start

```python
import sys
import os
sys.path.insert(0, 'src')  # Add src directory to path

from g2c import Console, GameState, InputHandler, print_header

# Create a simple game
def simple_game():
    print_header("My Game")
    
    state = GameState()
    state.set('score', 0)
    
    name = input("Enter your name: ")
    choice = InputHandler.get_choice("Choose (a/b/c): ", ["a", "b", "c"])
    number = InputHandler.get_number("Enter a number (1-10): ", 1, 10)
    
    print(f"Hello {name}! You chose {choice} and number {number}")

simple_game()
```

## Examples

See the `examples/` directory for complete example applications:

- `number_game.py` - Basic number guessing game
- `advanced_game.py` - Advanced game using full G2C framework features

## Requirements

- Python 3.6 or higher
- No external dependencies required

## License

MIT License - see LICENSE file for details.