import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load wine dataset
wine = load_wine()

# Create DataFrame
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['Target'] = wine.target

# Input and output
X = data.drop('Target', axis=1)
y = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Display confusion matrix using matplotlib
plt.figure(figsize=(7, 5))
plt.imshow(cm)
plt.colorbar()

plt.xticks(range(3), wine.target_names)
plt.yticks(range(3), wine.target_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KNN Confusion Matrix")

# Display values inside the matrix
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()
