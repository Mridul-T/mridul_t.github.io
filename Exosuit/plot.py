import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file
data = np.loadtxt("data.txt", skiprows=1, delimiter=",")
print(data.shape)
# Plot the data
plt.figure(figsize=(16,8))
plt.plot(data[400:1200, 0], data[400:1200, 1], label="")

# Add a title and labels to the axes

plt.title("Walking Slow")
plt.xlabel("x")
plt.ylabel("y")

# Show the plot
plt.show()