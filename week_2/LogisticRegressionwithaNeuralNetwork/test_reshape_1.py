import numpy as np

# Create a dataset of 5 RGB images of size 4x4 pixels
X = np.random.randint(0, 256, (5, 4, 4, 3))  # Shape (5, 4, 4, 3)

# Flatten each image into a column vector
X_flatten = X.reshape(X.shape[0], -1).T  # Shape (48, 5)

print("Original shape:", X.shape)
print("Flattened shape:", X_flatten.shape)

