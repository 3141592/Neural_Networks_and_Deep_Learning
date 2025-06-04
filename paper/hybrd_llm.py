import numpy as np
import matplotlib.pyplot as plt

# Simulated quality gaps H(x) for 100 examples
np.random.seed(42)
H = np.random.normal(loc=0.0, scale=1.0, size=100)  # mean 0, std 1

# Range of threshold values t
t_values = np.linspace(-3, 2, 100)
probabilities = [(H > t).mean() * 100 for t in t_values]  # as percentages

# Plot
plt.plot(t_values, probabilities, label="Pr[H(x) > t]")
plt.axvline(0, color='red', linestyle='--')
plt.axhline((H > 0).mean() * 100, color='red', linestyle='--')

plt.xlabel("Quality Gaps Threshold (t)")
plt.ylabel("Examples %: Pr[H(x) > t]")
plt.title("Toy Example of Quality Gap Distribution")
plt.legend()
plt.grid(True)
plt.show()

