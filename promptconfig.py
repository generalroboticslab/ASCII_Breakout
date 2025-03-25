import json
from typing import List, Dict, Optional

class PromptConfigurator:
    def __init__(self):
        # Predefined prompt components
        self.components = {
            # Game-Specific Components
            "context": {
                "breakout": "You are a professional Atari 2600 Breakout player.",
                "space_invaders": "You are a professional Atari 2600 Space Invaders player.",
                # Add more game-specific contexts
            },
            "game_context": {
                "breakout": "Breakout uses a paddle to bounce a ball with the aim of breaking all the bricks.",
                "space_invaders": "Space Invaders involves defending against descending alien invaders by shooting them.",
                # Add more game-specific game contexts
            },
            "strategy": {
                "breakout": "Keep the center of the paddle lined up with the ball on the x-axis.",
                "space_invaders": "Strategically eliminate aliens while avoiding their projectiles.",
                # Add more game-specific strategies
            },
            "actions": {
                "breakout": (
                    "Your goal is to provide the optimal action in the given game state. Your possible actions are:\n"
                    "• <action> 0 </action> for NOOP (do nothing)\n"
                    "• <action> 1 </action> for LEFT\n"
                    "• <action> 2 </action> for RIGHT\n"
                    "Immediately output the numerical value for the chosen action between the <action> tags, with no additional text or formatting."
                ),
                "space_invaders": (
                    "Your goal is to provide the optimal action in the given game state. Your possible actions are:\n"
                    "• <action> 0 </action> for NOOP (do nothing)\n"
                    "• <action> 1 </action> for LEFT\n"
                    "• <action> 2 </action> for RIGHT\n"
                    "• <action> 3 </action> for SHOOT\n"
                    "Immediately output the numerical value for the chosen action between the <action> tags, with no additional text or formatting."
                ),
                # Add more game-specific action descriptions
            },
            "observations": {
                "breakout": (
                    "You will receive an ASCII character grid representation of the game board that includes:\n"
                    "• The paddle represented as '=======' \n"
                    "• The ball represented as 'O'\n"
                    "• Bricks represented as '|____|'\n"
                    "• Boundry represented as '#'\n"
                    "• And empty space represented as ' '\n"
                    "The game board spans [0-79, 0-24] characters with newlines dividing rows."
                ),
                "space_invaders": (
                    "You will receive an ASCII character grid representation of the game board that includes:\n"
                    "• Your ship represented as '^'\n"
                    "• Alien ships represented as 'W'\n"
                    "• Projectiles represented as '|'\n"
                    "• Boundry represented as '#'\n"
                    "• And empty space represented as ' '\n"
                    "The game board spans [0-79, 0-24] characters with newlines dividing rows."
                ),
                # Add more game-specific observation descriptions
            },
            
            # Universal Components (Reasoning Steps)
            "reasoning_steps": {
                "board_analysis": (
                    "Carefully examine the entire ASCII board to determine the exact grid locations of key features. "
                    "Count the number of spaces if needed to locate exact positions. Take as long as you need to COMPLETELY "
                    "understand the board and key information. Double-check your counting to make you know the exact locations of key features."
                ),
                "context_consideration": (
                    "Take into account the broader game context provided by the board layout and locations of key features. "
                    "Take note of what features are present and where exactly they are relative to each other."
                ),
                "state_reporting": (
                    "Output the current game state exactly as observed."
                ),
                "step_by_step_reasoning": (
                    "Think through the board step by step to determine the best possible action given the game's objective, "
                    "while considering the full game context. Enclose your internal reasoning (your scratchpad) within "
                    "<reasoning> and </reasoning> meta tags. This reasoning is for your internal process only."
                ),
                "action_recommendation": (
                    "Based on your analysis, choose the optimal action in the given state."
                )
            }
        }

    def build_prompt(
        self, 
        game: str, 
        include_universal_steps: List[str] = None,
        custom_instructions: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Build a prompt with configurable components.
        
        :param game: Name of the game (e.g., 'breakout')
        :param include_universal_steps: List of universal reasoning steps to include
        :param custom_instructions: Dictionary of custom component overrides
        :return: Constructed prompt string
        """
        # Start with game-specific components
        prompt_parts = []
        
        # Add context
        if custom_instructions and 'context' in custom_instructions:
            prompt_parts.append(custom_instructions['context'])
        else:
            prompt_parts.append(self.components['context'].get(game, "You are a professional game player."))
        
        # Add game context
        if custom_instructions and 'game_context' in custom_instructions:
            prompt_parts.append(custom_instructions['game_context'])
        else:
            prompt_parts.append(self.components['game_context'].get(game, ""))
        
        # Add strategy
        if custom_instructions and 'strategy' in custom_instructions:
            prompt_parts.append(custom_instructions['strategy'])
        else:
            prompt_parts.append(self.components['strategy'].get(game, ""))
        
        # Add actions
        if custom_instructions and 'actions' in custom_instructions:
            prompt_parts.append(custom_instructions['actions'])
        else:
            prompt_parts.append(self.components['actions'].get(game, ""))
        
        # Add observations
        if custom_instructions and 'observations' in custom_instructions:
            prompt_parts.append(custom_instructions['observations'])
        else:
            prompt_parts.append(self.components['observations'].get(game, ""))
        
        # Add universal reasoning steps
        if include_universal_steps and len(include_universal_steps)>0:
            universal_steps = []
            for step in include_universal_steps:
                if step in self.components['reasoning_steps']:
                    universal_steps.append(
                        f"{step.replace('_', ' ').title()}: "
                        f"{self.components['reasoning_steps'][step]}"
                    )
            
            if universal_steps:
                prompt_parts.append("Your responsibilities are as follows:")
                prompt_parts.extend(f"{i+1}. {step}" for i, step in enumerate(universal_steps))
        
        return "\n".join(prompt_parts)

    def save_prompt_configuration(self, prompt: str, filename: str):
        """Save a prompt configuration to a file."""
        with open(filename, 'w') as f:
            f.write(prompt)

    def load_prompt_configuration(self, filename: str) -> str:
        """Load a prompt configuration from a file."""
        with open(filename, 'r') as f:
            return f.read()