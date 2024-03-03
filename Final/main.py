import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

positions = pd.read_csv(r"C:\Prog\rank.csv")

# Assuming 'positions' is your DataFrame and 'player_ranking_points' is the column
plt.hist(positions['tourneys_played'], bins=50, edgecolor='black')
plt.title('Player Tournaments Played')
plt.xlabel('Tournaments Played')
plt.ylabel('Number of Players')
plt.show()
