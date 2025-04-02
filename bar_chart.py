import matplotlib.pyplot as plt
import numpy as np

# Data
experiments = ["aster-r","aster-sm", "aster-med", "galax-r", "galax-sm", "galax-med"]
trial_1 = [-1570+3000, -570+3000, -170+3000, -600+3000, -310+3000, -140+3000]  # Replace with actual values
trial_2 = [-2970+3000, -320+3000, -470+3000, -560+3000, 110+3000, -120+3000]  # Replace with actual values

x = np.arange(len(experiments))  # Label locations
width = 0.4  # Width of bars

# Create figure and axes
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, trial_1, width, label='Trial 1', color='skyblue')
rects2 = ax.bar(x + width/2, trial_2, width, label='Trial 2', color='orange')

# Labels and title
ax.set_xlabel("Experiments")
ax.set_ylabel("Score")
ax.set_title("Scores over 1000 timesteps")
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()

# Show plot
plt.show()