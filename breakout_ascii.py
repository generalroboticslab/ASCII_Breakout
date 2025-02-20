from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import login
from gymnasium.spaces import Text
from openai import OpenAI 
import gymnasium as gym
import numpy as np
import imageio
import random
import torch
import time
import json 
import csv
import os
import re
from tqdm import tqdm

# Disable tokenizer parallelism warning.
# Do to fix warning wtih generating video.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Create Gymnasium Environment 
class BrickBreakerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    WIN_WIDTH = 80
    WIN_HEIGHT = 24
    INTERIOR_WIDTH = WIN_WIDTH - 2
    INTERIOR_HEIGHT = WIN_HEIGHT - 2
    BRICK_ROWS = 7
    BRICK_COLS = 13
    BRICK_WIDTH = 6
    BRICK_HEIGHT = 1
    GAP_ABOVE_BRICKS = 4
    GAP_BETWEEN_BRICKS_AND_PADDLE = 10

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        max_length = self.WIN_HEIGHT * (self.WIN_WIDTH + 1)
        self.observation_space = Text(max_length=max_length)
        self.action_space = gym.spaces.Discrete(3)
        self.paddle_speed = 2
        self.paddle_width = 7
        self.brick_start_y = 1 + self.GAP_ABOVE_BRICKS
        self.paddle_y = (self.brick_start_y + self.BRICK_ROWS) + self.GAP_BETWEEN_BRICKS_AND_PADDLE

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.score = 0
        self.lives = 3
        self.bricks = np.ones((self.BRICK_ROWS, self.BRICK_COLS), dtype=np.int32)
        self.ball_x = self.WIN_WIDTH // 2
        self.ball_y = self.paddle_y - 1
        self.ball_dx = random.choice([-1, 1])
        self.ball_dy = -1
        self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        return self._get_ascii(), {}

    def _get_ascii(self):
        grid = [[" ," for _ in range(self.WIN_WIDTH)] for _ in range(self.WIN_HEIGHT)]
        for x in range(self.WIN_WIDTH):
            grid[0][x] = "#,"
            grid[self.WIN_HEIGHT - 1][x] = "#,"
        for y in range(self.WIN_HEIGHT):
            grid[y][0] = "#,"
            grid[y][self.WIN_WIDTH - 1] = "#,"
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                if self.bricks[i, j] == 1:
                    brick_x = 1 + j * self.BRICK_WIDTH
                    brick_y = self.brick_start_y + i
                    for bx in range(self.BRICK_WIDTH):
                        ch = "B," if bx == 0 or bx == self.BRICK_WIDTH - 1 else "B,"
                        if brick_x + bx < self.WIN_WIDTH - 1:
                            grid[brick_y][brick_x + bx] = ch
        for i in range(self.paddle_width):
            if 0 <= self.paddle_x + i < self.WIN_WIDTH - 1:
                grid[self.paddle_y][self.paddle_x + i] = "=,"
        bx = int(round(self.ball_x))
        by = int(round(self.ball_y))
        if 0 <= by < self.WIN_HEIGHT and 0 <= bx < self.WIN_WIDTH:
            grid[by][bx] = "O,"
        return "\n".join("".join(row) for row in grid)

    def step(self, action):
        if action == 1:
            self.paddle_x = max(1, self.paddle_x - self.paddle_speed)
        elif action == 2:
            self.paddle_x = min(self.WIN_WIDTH - 1 - self.paddle_width, self.paddle_x + self.paddle_speed)
        reward = 0
        new_ball_x = self.ball_x + self.ball_dx
        new_ball_y = self.ball_y + self.ball_dy
        if new_ball_x <= 0:
            new_ball_x = 0
            self.ball_dx = -self.ball_dx
        elif new_ball_x >= self.WIN_WIDTH - 1:
            new_ball_x = self.WIN_WIDTH - 1
            self.ball_dx = -self.ball_dx
        if new_ball_y <= 0:
            new_ball_y = 0
            self.ball_dy = -self.ball_dy
        elif new_ball_y >= self.WIN_HEIGHT - 1:
            self.lives -= 1
            if self.lives <= 0:
                return self._get_ascii(), 0, True, False, {"score": self.score}
            new_ball_x = self.WIN_WIDTH // 2
            new_ball_y = self.paddle_y - 1
            self.ball_dx = random.choice([-1, 1])
            self.ball_dy = -1
            self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        if int(new_ball_y) == self.paddle_y and self.paddle_x <= int(new_ball_x) < self.paddle_x + self.paddle_width:
            new_ball_y = self.paddle_y - 1
            self.ball_dy = -abs(self.ball_dy)
            hit_offset = (new_ball_x - self.paddle_x) - (self.paddle_width / 2)
            self.ball_dx = 1 if hit_offset >= 0 else -1
        brick_row = int(new_ball_y) - self.brick_start_y
        if 0 <= brick_row < self.BRICK_ROWS:
            for j in range(self.BRICK_COLS):
                brick_x = 1 + j * self.BRICK_WIDTH
                brick_y = self.brick_start_y + brick_row
                if (self.bricks[brick_row, j] == 1 and 
                    brick_y == int(new_ball_y) and 
                    brick_x <= int(new_ball_x) < brick_x + self.BRICK_WIDTH):
                    self.bricks[brick_row, j] = 0
                    self.ball_dy = -self.ball_dy
                    reward += 10
                    self.score += 10
                    break
        self.ball_x = new_ball_x
        self.ball_y = new_ball_y
        done = (np.sum(self.bricks) == 0)
        if done:
            reward += 50
        return self._get_ascii(), reward, done, False, {"score": self.score}

    def render(self, mode="human"):
        ascii_obs = self._get_ascii()
        if mode == "human":
            print(ascii_obs)
            print(f"Score: {self.score}  Lives: {self.lives}")
        return ascii_obs

    def close(self):
        pass

def extract_action(output_decode: str) -> int:
    pattern = r"<action>\s*([012])\s*</action>"
    matches = re.findall(pattern, output_decode, flags=re.IGNORECASE)
    if not matches:
        raise ValueError("No valid <action> tag found in the output.")
    return int(matches[-1])

def ascii_to_image(ascii_text: str, num_cols=80, num_rows=24) -> Image.Image:
    """
    Converts an ASCII art string into a properly formatted image using a default monospaced font.
    """
    font = ImageFont.load_default()
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    img_width, img_height = num_cols * char_width, num_rows * char_height

    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)
    
    lines = ascii_text.splitlines()
    for row, line in enumerate(lines[:num_rows]):
        draw.text((0, row * char_height), line.ljust(num_cols), font=font, fill="white")
    
    return image

if __name__ == "__main__":

    api_value = input("Do you want to run OpenAI GPT-4o (1) or Llama 3.2 3B (2)? ")

    if api_value == str(1):
        print("\nSetting up GPT-4o...\n")
        openai_api_file = open("OPENAI_API_KEY.txt", "r") 
        openai_api_key = openai_api_file.read() 
        client = OpenAI(api_key=openai_api_key) 

    else:
        print("\nSetting up Llama 3.2 3B...\n")
        hugging_face_api_file = open("HG_API_KEY.txt", "r") 
        hugging_face_api_key = hugging_face_api_file.read() 

        login(token=hugging_face_api_key)

        set_seed(42)
        device = get_device()
        
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )

    print("\nModel set up complete...\n")
    
    messages = [{"role": "system", "content": f"You are a professional Atari 2600 game playing assistant and will be provided an ASCII representation of Breakout. "
                    "Breakout uses a paddle to hit a ball with the aim of breaking all bricks. Your goal is to provide me with"
                    "the best action I could take to break all the bricks while hitting the ball with the paddle, I only have control of the paddle and "
                    "can only move it left or right. You will receive observations in a 80 by 24 grid, extending [0-79, 0-24] x and y, where:"
                    "'#' represents the border of the game, 'B' represents a brick cell, '=' represents part of your paddle, '0' represents the ball, and ' ' represents empty space."
                    "Take as long as you need to COMPLETELY understand the board, and location of the paddle and ball toshow me the exact game state we are in."
                    "Double check your work to find the exact coordinate locations of the ball and paddle, this is essential. Make sure your calculations are correct, and redo them EVENY step."
                    "Once you have done this take that information and think step by step what the best action you could "
                    "take to keep the paddle in line with the ball. The potential actions I can take are <action> 0 </action> for NOOP, <action> 1 </action> for LEFT and "
                    "<action> 2 </action> for RIGHT. Provide output as a where your reasoning is between <reasoning> meta tags, this is a scratchpad for you to think, take as long as you need. "
                    "In addition, please add the numerical value for the action inbetween <action> </action> meta tags."
         }]

    env = BrickBreakerEnv(render_mode="human")
    
    obs, _ = env.reset(seed=42)

    # Record the initial frame.
    frames = []
    frames.append(ascii_to_image(obs, num_cols=160, num_rows=48))

    total_rewards = 0
    cumulative_rewards = []
    action_list = []

    # Should be number of inputs*2 + 1
    max_message_len = 7 # 3 input messages 
    with open('./all_responses.txt', "w") as file:
        file.write('')
    file.close()
    # Run for 1000 actions
    # Replicate Atari-GPT
    for i in tqdm(range(200), desc="Testing"):
        action = None
        # Give 3 chances at providing a correct action
        # If a correct action is given then break
        for _ in range(3):
            
            messages.append({
                    "role": "user",
                    "content": 
                            f"This is the current game state:\n{obs}\n\n"
                            "First, spatially interpret your observations and provide a summary of the current game state, including the exact position of key information such as your paddle and the ball."
                            "Based on the provided game state, please determine the best action to take. Remember:\n"
                            "- Provide your detailed reasoning between <reasoning> and </reasoning> meta tags.\n"
                            "- Then, immediately provide your chosen action between <action> and </action> meta tags, with no extra text or formatting."
                            "Remember your action output should only be one of these 3 options <action> 0 </action> for NOOP, <action> 1 </action> for LEFT and <action> 2 </action> for RIGHT\n"
                            "Show me what the current game state is:\n"
                        }
                    )
            
            if api_value == str(1):
                response = client.chat.completions.create(
                    model="o3-mini-2025-01-31",
                    messages=messages,
                    temperature=1,
                )

                output_decode = response.choices[0].message.content
            
            else:
                input_ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", return_dict=True
                ).to(get_device())

                outputs = model.generate(**input_ids, max_new_tokens=4096)

                output_decode = tokenizer.decode(outputs[0])

            # Write the message to a text file 
            with open('./all_responses.txt', "a", encoding='utf-8') as file:
                file.write(str(output_decode) + '\n\n')


            try:
                action = extract_action(output_decode)
                print(f"Extracted action: {action}")
                break

            except ValueError as e:
                print("Extraction error:", e)
                messages.append({
                    "role": "user",
                    "content":  "You did not put the action between the <action> and </action> meta tags. PROVIDE THE ENDING ACTION ONLY BETWEEN THESE META TAGS."
                })

        messages.append({
            "role": "assistant",
            "content": output_decode
        })

        action_list.append(action)

        obs, reward, done, truncated, info = env.step(action)

        total_rewards += reward
        cumulative_rewards.append(total_rewards)

        frames.append(ascii_to_image(obs, num_cols=160, num_rows=48))

        if len(messages) >= max_message_len:
            # pop the user and assistant message FIFO
            # use index 1 because of system prompt
            messages.pop(1)
            messages.pop(1)

        if done:
            obs, _ = env.reset(seed=42)
            frame_img = ascii_to_image(obs)
            frames.append(frame_img)
            done = False
        
        print("Step ", i)

        time.sleep(0.1)

    print("\n\n Total Reward: ", total_rewards)

    print("Saving video of performance...")
    video_filename = "breakout.mp4"
    images = [np.array(frame) for frame in frames if frame is not None]
    imageio.mimwrite(video_filename, images, fps=env.metadata["render_fps"])
    print(f"Saved video as {video_filename}")

    print("Saving actions and cumulative rewards...")

    header = ["actions", "cumulative_rewards"]

    with open('./actions_rewards.csv', 'w') as f:
          writer = csv.writer(f)
          writer.writerow(header)
          
          for action, cum_reward in zip(action_list, cumulative_rewards):
              writer.writerow([action, cum_reward])
    
    print("\nTest complete, Thank you!\n")