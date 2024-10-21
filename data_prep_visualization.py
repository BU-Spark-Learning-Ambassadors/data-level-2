# 1. Data Preparation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
data = pd.read_csv('-----')  # Fill in the file name or path (e.g., 'iris.csv')

# Check the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill missing values (if applicable)
# In the case of the Iris dataset, you might not have missing values, but this is good practice.
data.fillna(data['-----'].median(), inplace=True)  # If needed, specify the column for median filling (e.g., 'sepal_length')

# Detect outliers using box plot (e.g., sepal length)
plt.figure(figsize=(8, 6))
sns.boxplot(data['-----'])  # Replace with the column you want to check for outliers (e.g., 'sepal_length')
plt.title('----- Box Plot')  # Add a title for the box plot (e.g., 'Sepal Length Box Plot')
plt.show()

# 2. Data Transformation (Scaling features like 'sepal_length' and 'petal_length')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['-----', '-----']] = scaler.fit_transform(data[['-----', '-----']])  # Scale 'sepal_length' and 'petal_length'

# 3. Data Visualization

# Histogram for Sepal Length
plt.figure(figsize=(8, 6))
plt.hist(data['-----'], bins=30)  # Specify the column you want to visualize (e.g., 'sepal_length')
plt.title('----- Distribution')  # Add a title for the histogram (e.g., 'Sepal Length Distribution')
plt.xlabel('-----')  # Label the x-axis (e.g., 'Sepal Length')
plt.ylabel('Frequency')  # Label the y-axis
plt.show()

# Scatter Plot (e.g., Sepal Length vs. Petal Length)
plt.figure(figsize=(8, 6))
plt.scatter(data['-----'], data['-----'])  # Specify the x (e.g., 'sepal_length') and y (e.g., 'petal_length') columns
plt.title('----- vs. -----')  # Title for the scatter plot (e.g., 'Sepal Length vs. Petal Length')
plt.xlabel('-----')  # Label the x-axis (e.g., 'Sepal Length')
plt.ylabel('-----')  # Label the y-axis (e.g., 'Petal Length')
plt.show()

# Heatmap of Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')  # Heatmap for correlation of numerical features
plt.title('Correlation Matrix of Iris Dataset')
plt.show()
