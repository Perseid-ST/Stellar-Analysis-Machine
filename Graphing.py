import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the two CSV files
file1 = "trial1.csv"
file2 = "ClusterProPlus.csv"
data1 = pd.read_csv(file1) #from trial1.csv
data2 = pd.read_csv(file2) #from Cluster Pro Plus.csv

# Extract columns for comparison
#distance
dm1 = data1['distance_kpc']
dm_error1 = data1['distance_kpc_error']  
cpp_distance2 = data2['CPP Distance'] 

#age
cpp_loga2 = data2['CPP Log(Age)']
loga1 = data1['loga']
loga_error1 = data1['loga _error']

#metallicity
cpp_met2 = data2['CPP Metallicity']
met1 = data1['met']
met_error1 = data1['met  _error']

#extinction
cpp_ebv2 = data2['CPP E(B-V)']
av1 = data1['E_BV']
av_error1 = data1['E_BV_error']


# Function to plot comparisons
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
    best_fit_label = f"Best Fit Line: y = {slope:.3f}x + {intercept:.3f}"
    plt.plot(x, slope * x + intercept, color='green', label=best_fit_label)

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
                "Distance Comparison", "SAM Distance (kpc)", "CPP Distance", "distance_comparison1.png")

plot_comparison(loga1, cpp_loga2, loga_error1,
                "Log(Age) Comparison", "SAM Age (Log(Age))", "CPP Log(Age)", "loga_comparison1.png")

plot_comparison(met1, cpp_met2, met_error1,
                "Metallicity Comparison", "SAM Metallicity", "CPP Metallicity", "metallicity_comparison1.png")

plot_comparison(av1, cpp_ebv2, av_error1, 
                "Extinction Comparison", "SAM E_BV", "CPP E(B-V)", "extinction_comparison1.png")