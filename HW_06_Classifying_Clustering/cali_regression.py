# Mahdy Karam
# Feb 9, 2024

from statistics import linear_regression
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load California housing dataset and display its description, shape, and feature names
california = fetch_california_housing()
print(california.DESCR)
print(california.data.shape)
print(california.feature_names)

# Explore the dataset using Pandas
# Create a dataframe
california_df = pd.DataFrame(california.data,
                            columns=california.feature_names)

# Add the labels of median housing values
california_df['MedHouseValue'] = pd.Series(california.target)

# Print the top few rows
print(california_df.head())

# Print some summary statistics
print(california_df.describe())

# Visualize data with Seaborn
# Get a sample to graph
sample_df = california_df.sample(frac=0.1, random_state=17)
sns.set(font_scale=2)
sns.set_style('whitegrid')
for feature in california.feature_names:
    plt.figure(figsize=(16,9))
    sns.scatterplot(data=sample_df, 
                    x=feature,
                    y='MedHouseValue',
                    hue='MedHouseValue',
                    palette='cool',
                    legend=False)
    plt.show()

# Perform linear regression
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(california.data,
                                                    california.target,
                                                    random_state=11)
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

# Print coefficients for multiple linear regression
# One coefficient for each feature and an intercept
for i, name in enumerate(california.feature_names):
    print(f'{name:>10}: {linear_regression.coef_[i]}')

# Predict using the trained model
predicted = linear_regression.predict(X_test)
expected = y_test

# Visualize predicted vs. expected values
# Set up the data in the format required by Pandas
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)

figure = plt.figure(figsize=(9,9))
axes = sns.scatterplot(data=df, x='Expected', y='Predicted',
                        hue='Predicted', palette='cool', legend=False)

# Set up axes
start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
axes.set_xlim(start,end)
axes.set_ylim(start,end)

# Add a line representing perfect prediction
line = plt.plot([start,end], [start,end], 'k--')
plt.show()

# Evaluate the model using metrics
# Coefficient of determination (r-squared)
# A score of 1.0 indicates perfect prediction, 0.0 indicates no predictive power
print(metrics.r2_score(expected, predicted))

# Mean squared error (squared error averaged)
# Lower values indicate better performance
print(metrics.mean_squared_error(expected, predicted))
