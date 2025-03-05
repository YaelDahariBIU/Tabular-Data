import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example Data (Replace with actual similarity scores)
data = {
    'Dataset': ['Movies Revenue', 'Laptop Price', 'Cars Price', 'Avocado Average Price'],
    'Random': [0.511, 0.808, 0.871, 0.787],
    'Mean': [0.923, 0.920, 0.951, 0.856],
    'Median': [0.937, 0.925, 0.951, 0.861],
    'Frequent': [0.934, 0.920, 0.947, 0.835],
    'KNN': [0.923, 0.920, 0.951, 0.856],
    'LR': [1.000, 1.00, 1.00, 1.00]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set figure size
plt.figure(figsize=(10, 6))

# Define number of datasets & methods
num_datasets = len(df)
num_methods = len(df.columns) - 1  # Excluding 'Dataset' column
bar_width = 0.12  # Width of bars

# Set positions for bars
x = np.arange(num_datasets)

# Use seaborn color palette
colors = sns.color_palette("Accent", num_methods)

# Plot bars for each method
for i, method in enumerate(df.columns[1:]):  # Skip 'Dataset' column
    plt.bar(x + i * bar_width, df[method], width=bar_width, label=method, color=colors[i])

# Improve readability
plt.xticks(x + bar_width * (num_methods / 2), df['Dataset'])  # Center labels
plt.ylabel("Average Similarity Score")
plt.xlabel("Dataset")
plt.title("Comparison of Imputation Methods (Average Similarity Score per Dataset)")
plt.ylim(0.4, 1.05)  # Keep range reasonable for better visualization
plt.legend(title="Imputation Method", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()  # Adjust layout to make room for the legend

# Save plot
plt.savefig('avg_sim_per_ds.png', dpi=300)  # Save before showing to avoid blank image

# Show plot
plt.show()
