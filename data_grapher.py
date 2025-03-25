import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# Get all CSV files in subfolders of 'logs/'
file_paths = glob.glob("logs/*/*.csv")

# Define pattern to match model names
pattern = re.compile(r".*4o.*")  # Modify this regex pattern as needed

# Dictionary to store data
data = {}

# Load CSV files
for file in file_paths:
    model_name = os.path.basename(os.path.dirname(file))  # Extract model name from parent folder
    if pattern.match(model_name):
        df = pd.read_csv(file)
        data[model_name] = df

# Define marker styles
markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']

# Plot cumulative rewards for selected models
plt.figure(figsize=(10, 6))

for i, (model, df) in enumerate(data.items()):
    plt.plot(df.index, df["cumulative_rewards"], label=model, alpha=0.7, marker=markers[i % len(markers)], markevery=50)

plt.xlabel("Time Step")
plt.ylabel("Cumulative Rewards")
plt.title("Cumulative Rewards Over Time for Selected Models")
plt.legend()
plt.grid(True)
plt.show()