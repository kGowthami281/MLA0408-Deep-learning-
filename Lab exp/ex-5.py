import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Iris dataset
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display first 5 rows
print(data.head())

# Input and output
X = data[['sepal length (cm)']]
y = data['sepal width (cm)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test values
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))

# Plot actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, marker='*', label='Actual')
plt.plot(X_test, y_pred, linewidth=3, label='Predicted')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Linear Regression: Sepal Width vs Sepal Length')
plt.legend()
plt.show()

# Predict for a new sample
new_sample = pd.DataFrame(
    [[5]], columns=['sepal length (cm)']
)

predicted_width = model.predict(new_sample)

print(
    f"Predicted sepal width for sepal length 5 cm: "
    f"{predicted_width[0]:.2f} cm"
)
