'''EPS Advanced Topics in CS Winter 23-24 Final Project'''
# Mahdy M. Karam
# 2024/03/01

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Import data
data = pd.read_csv(r"C:\Prog\rank.csv")

# Drop columns we don't need
columns_to_drop = ['move_positions', 'move_direction',
                   'week_title', 'week_year', 'week_month', 'week_day']
data = data.drop(columns=columns_to_drop)

# Remove rows with NaN values anywhere
data = data.dropna()

# Define features and target
X = data[[ 'player_age', 'ranking_points', 'tourneys_played']]
y = data['rank_number']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)
knn_mse = mean_squared_error(y_test, knn_predictions)
print("KNN Mean Squared Error:", knn_mse)

# K-Means Clustering
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X_train_scaled)
kmeans_predictions = kmeans_model.predict(X_test_scaled)
kmeans_mse = mean_squared_error(y_test, kmeans_predictions)
print("K-Means Mean Squared Error:", kmeans_mse)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)
linear_mse = mean_squared_error(y_test, linear_predictions)
print("Linear Regression Mean Squared Error:", linear_mse)
