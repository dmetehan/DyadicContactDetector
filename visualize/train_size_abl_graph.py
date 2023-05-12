import matplotlib.pyplot as plt

x = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
# accuracy = [80.01, 83.44, 79.63, 77.98, 80.77, 80.14, 79.19, 78.17, 79.31, 76.59]
# balanced_accuracy = [79.60, 83.23, 80.00, 78.39, 81.66, 80.07, 78.59, 78.96, 77.66, 75.25]
# f1_score = [76.33, 80.54, 77.15, 75.44, 79.12, 77.04, 75.08, 76.24, 73.19, 70.55]
accuracy = [80.01, 75.95, 79.00, 80.90, 77.09, 74.18, 78.05, 75.19, 72.97, 75.00]
balanced_accuracy = [79.60, 77.11, 78.71, 79.77, 78.48, 74.84, 79.26, 75.48, 71.33, 73.12]
f1_score = [76.33, 74.55, 75.39, 76.13, 76.05, 71.87, 76.75, 72.25, 65.48, 67.33]

plt.plot(x, accuracy, label="Accuracy", marker="o")
plt.plot(x, balanced_accuracy, label="Balanced Accuracy", marker="o")
plt.plot(x, f1_score, label="F1 Score", marker="o")

plt.xlabel("Percentage of training data (%)")
plt.ylabel("Performance Metrics (%)")
plt.title("Ablation Study Visualization")
plt.legend()

plt.gca().invert_xaxis()
#plt.show()
plt.savefig('foo.png', bbox_inches='tight')

