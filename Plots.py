import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load variables from the file
is_slippery_variables = np.load('is_slippery_variables.npy', allow_pickle=True)
non_slippery_variables = np.load('non_slippery_variables.npy', allow_pickle=True)

# Access the variable my_array
q_table = is_slippery_variables.item().get('q_table')
q_table2 = non_slippery_variables.item().get('q_table')
rewards_per_thousand_episodes = is_slippery_variables.item().get('rewards_per_thousand_episodes')
rewards_per_thousand_episodes2 = non_slippery_variables.item().get('rewards_per_thousand_episodes')


count = 1000
a = []
c = ["slip","slip","slip","slip","slip","slip","slip","slip","slip","slip","non-slip","non-slip","non-slip","non-slip","non-slip","non-slip","non-slip","non-slip","non-slip","non-slip",]
b = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
print(a)
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    ans = sum(r) / 1000
    print(count, ": ", ans)
    a.append(ans)
    count += 1000

for r in rewards_per_thousand_episodes2:
    ans = sum(r) / 1000
    print(count, ": ", ans)
    a.append(ans)
    count += 1000

print("\n\n********Q-table********\n")
print(q_table)

dict = {"Episodes":b, 'Rewards Per Episode': a, "slip":c}

df = pd.DataFrame(dict)
print(df)

#creates heatmap for q_table


first = sns.heatmap(q_table, annot=True, xticklabels=["left", "down", "right", "up"],yticklabels=["1", "2", "3","4", "5", "6","7", "8", "9","10", "11", "12","13", "14", "15","16"], vmax=1.0)
first.xaxis.tick_top()
plt.savefig(fname="q_table1.png", dpi=300)
plt.show()

second = sns.heatmap(q_table2, annot=True, xticklabels=["left", "down", "right", "up"],yticklabels=["1", "2", "3","4", "5", "6","7", "8", "9","10", "11", "12","13", "14", "15","16"], vmax=1.0)
second.xaxis.tick_top()
plt.savefig(fname="q_table2.png", dpi=300)
plt.show()

# Set max y-axis value to 1.0
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Episodes", y="Rewards Per Episode", hue="slip")
plt.ylim(0, 1.0)  # Set the y-axis limit to 0 to 1.0
plt.xlim(0, 10000)  # Set the y-axis limit to 0 to 1.0
plt.xlabel("Episodes")
plt.ylabel("Average Reward Per Episode")
plt.grid(True)
plt.savefig(fname="lineplot2.png", dpi=300)
plt.show()

game_board = np.array(
    [
        [0.75, 0, 0, 0],
        [0, 0.25, 0, 0.25],
        [0, 0, 0, 0.25],
        [0.25, 0, 0, 1.25]
    ]
)

# Create subplots for each choice
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Define action labels
actions = ['←', '↓', '→', '↑']

# Reshape the Q-tables into 4x4 grids
q_table_4x4 = np.argsort(q_table, axis=1)[:, ::-1][:, :4].reshape(4, 4, 4)
q_table2_4x4 = np.argsort(q_table2, axis=1)[:, ::-1][:, :4].reshape(4, 4, 4)

# Define game boards
game_board = np.array(
    [
        [0.75, 0, 0, 0],
        [0, 0.25, 0, 0.25],
        [0, 0, 0, 0.25],
        [0.25, 0, 0, 1.25]
    ]
)

# Plotting for q_table
for choice in range(2):
    ax = axes[0, choice]

    sns.heatmap(
        data=game_board,
        vmin=0,
        vmax=3,
        cmap='Paired',
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    # Add arrows to represent the actions
    for i in range(4):
        for j in range(4):
            action_index = q_table_4x4[i, j, choice]

            ax.text(
                x=j + 0.5,
                y=i + 0.5,
                s=actions[action_index],
                ha='center',
                va='center',
                size=20
            )

# Plotting for q_table2
for choice in range(2):
    ax = axes[1, choice]

    sns.heatmap(
        data=game_board,
        vmin=0,
        vmax=3,
        cmap='Paired',
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    # Add arrows to represent the actions
    for i in range(4):
        for j in range(4):
            action_index = q_table2_4x4[i, j, choice]

            ax.text(
                x=j + 0.5,
                y=i + 0.5,
                s=actions[action_index],
                ha='center',
                va='center',
                size=20
            )

plt.tight_layout()
plt.savefig(fname="gameboard.png", dpi=300)
plt.show()