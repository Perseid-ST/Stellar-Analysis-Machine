import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the two CSV files
file1 = "75 with adj ranges.csv"
file2 = "ClusterProPlus.csv"
data1 = pd.read_csv(file1)  # from trial1.csv
data2 = pd.read_csv(file2)  # from Cluster Pro Plus.csv

# Combine the two files into one DataFrame
# If there's a common key, use merge. Otherwise, use concat.
combined_data = pd.concat([data1, data2], axis=1)

# Filter out rows with NaN values in the relevant columns
filtered_data = combined_data.dropna(subset=['distance_kpc', 'distance_kpc_error', 'CPP Distance',
                                             'loga', 'loga _error', 'CPP Log(Age)',
                                             'met', 'met  _error', 'CPP Metallicity',
                                             'E_BV', 'E_BV_error', 'CPP E(B-V)'])

# Extract filtered columns for plotting
dm1 = filtered_data['distance_kpc']
dm_error1 = filtered_data['distance_kpc_error']
cpp_distance2 = filtered_data['CPP Distance']

loga1 = filtered_data['loga']
loga_error1 = filtered_data['loga _error']
cpp_loga2 = filtered_data['CPP Log(Age)']

met1 = filtered_data['met']
met_error1 = filtered_data['met  _error']
cpp_met2 = filtered_data['CPP Metallicity']

av1 = filtered_data['E_BV']
av_error1 = filtered_data['E_BV_error']
cpp_ebv2 = filtered_data['CPP E(B-V)']

# Function to plot comparisons
def plot_comparison(x, y, xerr, title, xlabel, ylabel, output_file):
    # Ensure the output folder exists
    output_folder = "comparisons"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Full path for the output file
    output_path = os.path.join(output_folder, output_file)

    plt.figure(figsize=(10, 6))
    # Plot data from the first CSV
    plt.errorbar(x, y, xerr=xerr, fmt='o', ecolor='red', capsize=3, label='SAM Data')

    # Calculate axis limits based on the data
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    # Add a y = x line
    min_val = max(x_min, y_min)  # Start the y = x line at the larger of x_min and y_min
    max_val = min(x_max, y_max)  # End the y = x line at the smaller of x_max and y_max
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='blue', label='CPP = SAM')

    # Calculate best fit line for my data
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept  # Predicted y values
    best_fit_label = f"Best Fit Line: y = {slope:.3f}x + {intercept:.3f}"
    plt.plot(x, y_pred, color='green', label=best_fit_label)

    # Calculate R² value
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    # Add the slope and R² as text on the graph
    text_x = x_min + (x_max - x_min)*0.01  # Position text slightly to the right of the minimum x value
    text_y = y_max + (y_max - y_max)*0.5 # Position text slightly below the maximum y value
    plt.text(text_x, text_y, f"R²: {r_squared:.3f}", fontsize=10, color='green', bbox=dict(facecolor='white', alpha=0.5))

    # Set axis limits based on the data
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add labels, title, and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


# Plot comparisons for each parameter
plot_comparison(dm1, cpp_distance2, dm_error1,
                "Distance Comparison", "SAM Distance (kpc)", "CPP Distance", "distance_comparison75.png")

plot_comparison(loga1, cpp_loga2, loga_error1,
                "Log(Age) Comparison", "SAM Age (Log(Age))", "CPP Log(Age)", "loga_comparison75.png")

plot_comparison(met1, cpp_met2, met_error1,
                "Metallicity Comparison", "SAM Metallicity", "CPP Metallicity", "metallicity_comparison75.png")

plot_comparison(av1, cpp_ebv2, av_error1,
                "Extinction Comparison", "SAM E_BV", "CPP E(B-V)", "extinction_comparison75.png")