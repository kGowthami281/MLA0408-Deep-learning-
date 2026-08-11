import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values
x = np.arange(-5, 5, 0.1)

# Calculate sigmoid values
y = sigmoid(x)

# Plot sigmoid function
plt.plot(x, y, color='pink')

plt.title('Visualization of the Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.grid()
plt.show()
