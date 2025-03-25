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
import argparse
from tqdm import tqdm
from comma_env import BrickBreakerCommaEnv
from no_comma_env import BrickBreakerEnv
from promptconfig import PromptConfigurator

import datetime

# Disable tokenizer parallelism warning.
# Do to fix warning wtih generating video.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_action(output_decode: str) -> int:
    pattern = r"<action>\s*([012])\s*</action>"
    matches = re.findall(pattern, output_decode, flags=re.IGNORECASE)
    if not matches:
        raise ValueError("No valid <action> tag found in the output.")
    return int(matches[-1])

def ascii_to_image(ascii_text: str, num_cols: int, num_rows: int) -> Image.Image:
    """
    Converts an ASCII art string into a properly formatted image using a monospaced font.
    Ensures all characters, including spaces, have consistent width.
    """
    # Use a monospaced font to ensure consistent character width
    from PIL import ImageFont
    import io

    # Create a monospaced font
    # Using a specific monospaced font that's widely available
    try:
        # Try to use a monospaced font that's likely to be available
        font = ImageFont.truetype("lucon.ttf", 16)  # Lucida Console
    except IOError:
        try:
            font = ImageFont.truetype("courbd.ttf", 16)  # Courier Bold
        except IOError:
            # Fallback to default if no specific monospaced font is found
            font = ImageFont.load_default()

    # Get character dimensions
    # Use a space character to measure width to ensure consistent spacing
    bbox = font.getbbox(" ")
    char_width = bbox[2] - bbox[0]
    
    # Measure character height (use 'A' to get a standard character height)
    bbox_height = font.getbbox("A")
    char_height = bbox_height[3] - bbox_height[1]

    # Calculate image dimensions
    img_width, img_height = num_cols * char_width, num_rows * char_height

    # Create image
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)
    
    # Process lines
    lines = ascii_text.splitlines()
    for row, line in enumerate(lines[:num_rows]):
        # Ensure each line is padded to full width with spaces
        padded_line = line.ljust(num_cols)
        
        # Draw each character individually to ensure consistent spacing
        for col, char in enumerate(padded_line):
            draw.text(
                (col * char_width, row * char_height), 
                char, 
                font=font, 
                fill="white"
            )
    
    return image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Atari-GPT evaluations')
    parser.add_argument('--critic', default=False,
                        help='Should the model use a critic?')
    parser.add_argument('--comma', default=False,
                        help='Do you want to use a comma?')
    args = parser.parse_args()

    # Create a timestamp and strings for naming files based on arguments.
    comma_str = "with_comma" if args.comma else "no_comma"
    critic_str = "with_critic" if args.critic else "no_critic"



    api_value = input("Do you want to run OpenAI GPT-4o (1),  OpenAI GPT-4o-mini (2), OpenAI o1-mini (3), OpenAI o3-mini (4), Llama 3.2 3B (5)?, or ALL (6)")

    if api_value == str(1):
        print("\nSetting up GPT-4o...\n")
        openai_api_file = open("OPENAI_API_KEY.txt", "r") 
        openai_api_key = openai_api_file.read() 
        client = OpenAI(api_key=openai_api_key) 
        models = ["gpt-4o-2024-08-06"]
    elif api_value == str(2):
        print("\nSetting up GPT-o1-mini...\n")
        openai_api_file = open("OPENAI_API_KEY.txt", "r") 
        openai_api_key = openai_api_file.read() 
        client = OpenAI(api_key=openai_api_key) 
        models = ["gpt-4o-mini-2024-07-18"]

    elif api_value == str(3):
        print("\nSetting up GPT-4o...\n")
        openai_api_file = open("OPENAI_API_KEY.txt", "r") 
        openai_api_key = openai_api_file.read() 
        client = OpenAI(api_key=openai_api_key) 
        models = ["o1-mini-2024-09-12"]

    elif api_value == str(4):
        print("\nSetting up GPT-4o...\n")
        openai_api_file = open("OPENAI_API_KEY.txt", "r") 
        openai_api_key = openai_api_file.read() 
        client = OpenAI(api_key=openai_api_key) 
        models = ["o3-mini-2025-01-31"]

    elif api_value == str(5):
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
        models = ["llama"]
    else:
        print("\nSetting up GPT...\n")
        openai_api_file = open("OPENAI_API_KEY.txt", "r") 
        openai_api_key = openai_api_file.read() 
        client = OpenAI(api_key=openai_api_key) 
        models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]

    print("\nModel set up complete...\n")
    tests_per_model = 2


    configurator = PromptConfigurator()
    breakout_small_prompt = configurator.build_prompt(
            game='breakout', 
            include_universal_steps=[
            ]
        )
    breakout_medium_prompt = configurator.build_prompt(
            game='breakout', 
            include_universal_steps=[
                'board_analysis', 
                'action_recommendation'
            ]
        )
    breakout_large_prompt = configurator.build_prompt(
            game='breakout', 
            include_universal_steps=[
                'board_analysis', 
                'context_consideration', 
                'step_by_step_reasoning', 
                'action_recommendation'
            ]
        )
    prompts = [breakout_small_prompt, breakout_medium_prompt, breakout_large_prompt]
    for m in range(len(models)):
        for p in range(len(prompts)):
            for test in range(tests_per_model):


                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                print("testing "+str(models[m]))



                messages = [{"role": "user", "content": prompts[p]
                    }]

                if comma_str == "with_comma":
                    env = BrickBreakerCommaEnv(render_mode="human")
                else:
                    env = BrickBreakerEnv(render_mode="human")
                
                obs, _ = env.reset(seed=42)

                # Record the initial frame.
                frames = []
                frames.append(ascii_to_image(obs, num_cols=80, num_rows=34))

                total_rewards = 0
                cumulative_rewards = []
                action_list = []

                # Should be number of inputs*2 + 1
                max_message_len = 3 # 3 input messages 
                responses_filename = f"./all_responses_{models[m]}_prompt_{p}_{comma_str}_{critic_str}_{timestamp}.txt"
                with open(responses_filename, "a") as file:
                    file.write('')
                file.close()
                # Run for 1000 actions
                # Replicate Atari-GPT
                for i in tqdm(range(500), desc="Testing"):
                    action = None
                    # Give 3 chances at providing a correct action
                    # If a correct action is given then break
                    for s in range(3):
                        success = False
                        try:
                            for _ in range(3):
                                
                                messages.append({
                                        "role": "user",
                                        "content": 
                                                f"""This is the current game state:
                                                {obs}

                                                Based on the provided game state, please determine the best action to take. Remember:
                                                - Provide your detailed reasoning between <reasoning> and </reasoning> meta tags.
                                                - Then, immediately provide your chosen action between <action> and </action> meta tags, with no extra text or formatting.

                                                Your action output should only be one of these 3 options:
                                                <action> 0 </action> for NOOP,
                                                <action> 1 </action> for LEFT,
                                                <action> 2 </action> for RIGHT."""
                                            }
                                        )
                                
                                
                                
                                if api_value == str(5):
                                    input_ids = tokenizer.apply_chat_template(
                                        messages, return_tensors="pt", return_dict=True
                                    ).to(get_device())

                                    outputs = model.generate(**input_ids, max_new_tokens=4096)

                                    output_decode = tokenizer.decode(outputs[0])
                                else:
                                    response = client.chat.completions.create(
                                        model=models[m],
                                        messages=messages,
                                        temperature=1

                                    )

                                    output_decode = response.choices[0].message.content

                                # Write the message to a text file 
                                with open(responses_filename, "a") as file:
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
                            success = True

                        except:
                            print("prompt/output error")
                            success = False
                        if success:
                            break
                    messages.append({
                        "role": "assistant",
                        "content": output_decode
                    })

                    action_list.append(action)

                    obs, reward, done, truncated, info = env.step(action)

                    total_rewards += reward
                    cumulative_rewards.append(total_rewards)

                    
                    img = ascii_to_image(obs, num_cols=80, num_rows=34)
                    frames.append(img)

                    if len(messages) >= max_message_len:
                        # pop the user and assistant message FIFO
                        # use index 1 because of system prompt
                        messages.pop(1)
                        messages.pop(1)

                    if done:
                        break
                        obs, _ = env.reset(seed=42)

                    
                    print("Step ", i)

                    time.sleep(0.1)

                print("\n\n Total Reward: ", total_rewards)

                print("Saving video of performance...")
                video_filename = f"breakout_{models[m]}_prompt_{p}_{comma_str}_{critic_str}_{timestamp}.mp4"
                images = [np.array(frame) for frame in frames if frame is not None]
                imageio.mimwrite(video_filename, images, fps=env.metadata["render_fps"])
                print(f"Saved video as {video_filename}")



                print("Saving actions and cumulative rewards...")
                header = ["actions", "cumulative_rewards"]
                csv_filename = f"./actions_rewards_{models[m]}_prompt_{p}_{comma_str}_{critic_str}_{timestamp}.csv"
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for action, cum_reward in zip(action_list, cumulative_rewards):
                        writer.writerow([action, cum_reward])
                
                print("\nTest complete, Thank you!\n")       