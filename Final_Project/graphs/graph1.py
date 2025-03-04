import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {
    'Attribute': ['budget', 'vote_count', 'popularity', 'original_language'],  # Attributes we imputed
    'Random': [0.448, 0.457, 0.743, 0.766],  # R² scores per method
    'Mean': [0.709, 0.670, 0.769, 0.767],
    'Median': [0.705, 0.660, 0.769, 0.767],
    'Frequent': [0.706, 0.645, 0.772, 0.767],
    'KNN': [0.709, 0.670, 0.769, 0.767],
    'LR': [0.694, 0.644, 0.771, 0.766],
    'Drop': [0.737, 0.765, 0.752, 0.781]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set up the figure
plt.figure(figsize=(10, 6))

# Define colors for each method
colors = ['red', 'brown', 'green', 'blue', 'purple', 'gray', 'orange']

# Loop through methods and plot each
for i, method in enumerate(df.columns[1:]):  # Skip 'Attribute' column
    plt.plot(df['Attribute'], df[method], marker='o', linestyle='-', label=method, color=colors[i])

# Improve readability
plt.xlabel("Imputed Attribute")
plt.ylabel("R² Score")
plt.title("Comparison of Imputation Methods based on R² Score in Movies Dataset")
plt.legend(title="Imputation Method")
plt.grid(True, linestyle='--', alpha=0.6)

# Show plot
# plt.show()

# Save plot
plt.savefig('imputation_comparison.png', dpi=300)