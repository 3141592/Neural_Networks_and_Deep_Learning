import tensorflow as tf

# True labels (one-hot)
y_true = [[0, 0, 1], [0, 1, 0]]
print(y_true)
# Predicted probabilities
y_pred = [[0.05, 0.10, 0.85], [0.1, 0.8, 0.1]]
print(y_pred)
loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
print(loss.numpy())  # e.g., [0.1625..., 0.2231...]

