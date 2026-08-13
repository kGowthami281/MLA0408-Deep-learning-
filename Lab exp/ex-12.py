import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add species column
data['Species'] = iris.target

# Separate features and target
X = data.drop('Species', axis=1)
y = data['Species']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1
)

# Create Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=1
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

plt.xticks(range(3), iris.target_names)
plt.yticks(range(3), iris.target_names)

# Add values inside the matrix
for i in range(3):
    for j in range(3):
        plt.text(
            j, i, conf_matrix[i, j],
            ha='center',
            va='center'
        )

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
