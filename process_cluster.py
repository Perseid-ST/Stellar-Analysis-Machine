import subprocess
import pandas as pd
import threading
import time
import os
import numpy as np
# This script processes a CSV file containing cluster names, runs isofitting.py for each cluster,

# Global flag to control the loading animation
loading = False

def loading_animation():
    """Display a loading animation (dots) in the terminal."""
    global loading
    while loading:
        for char in "|/-\\":
            print(f"\rProcessing clusters... {char}", end="", flush=True)
            time.sleep(0.2)
    print("\rProcessing clusters... Done!", flush=True)

def dm_to_kpc(dm):
    """Convert distance modulus (dm) to distance in kiloparsecs (kpc)."""
    return 10**((dm - 5) / 5)

def calculate_e_bv(av, rv=3.1):
    """
    Calculate color excess (E(B-V)) from total extinction (Av).

    :param av: Total extinction (Av) in magnitudes
    :param rv: Ratio of total to selective extinction (default is 3.1)
    :return: Color excess (E(B-V)) in magnitudes
    """
    return av / rv

def process_cluster(cluster_name):
    try:
        print(f"Processing cluster: {cluster_name}")
        # Check if the cluster name is empty or None
        if not cluster_name or pd.isna(cluster_name):
            print("Error: Cluster name is empty or None.")
            return None

        # Run isofitting.py with the cluster name as an argument and capture its output
        result = subprocess.run(
            ["python", "isofitting.py", cluster_name],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\nError processing cluster {cluster_name}: {e.stderr}")
        return None
    print(f"\nSuccessfully processed cluster: {cluster_name}")

    # Parse the output from isofitting.py
    output_lines = result.stdout.strip().split("\n")
    parameters = {}
    allowed_params = ["met", "loga", "dm", "Av", "met_error", "loga_error", "dm_error", "Av_error"]
    for line in output_lines:
        # Capture only lines matching the format "key: value" or "key_error: value"
        if any(param in line for param in allowed_params):
            key, value = line.split(":", 1)
            parameters[key.strip()] = value.strip()

            # Ensure only the relevant columns are returned
    relevant_keys = ["met", "met  _error", "loga", "loga _error", "dm", "dm   _error", "Av", "Av   _error"]
    filtered_parameters = {key: parameters.get(key, None) for key in relevant_keys}
    return filtered_parameters


def process_clusters_from_csv(csv_file):
    global loading
    try:
        # Start timing
        start_time = time.time()

        # Read the CSV file
        data = pd.read_csv(csv_file)
        # Ensure the column containing cluster names is named 'cluster_name'
        if 'cluster_name' not in data.columns:
            print("Error: CSV file must contain a 'cluster_name' column.")
            return

        # Create a list to store the results
        results = []
        # Start the loading animation in a separate thread
        loading = True
        animation_thread = threading.Thread(target=loading_animation)
        animation_thread.start()

        # Iterate over each cluster name and process it
        for cluster_name in data['cluster_name']:
            parameters = process_cluster(cluster_name)
            if parameters:
                parameters['cluster_name'] = cluster_name
                results.append(parameters)

        # Stop the loading animation
        loading = False
        animation_thread.join()

        # Convert results to a DataFrame and merge with the original data
        if results:
            results_df = pd.DataFrame(results)
            updated_data = pd.merge(data, results_df, on='cluster_name', how='left')

            # Apply dm_to_kpc to the 'dm' column and create a new 'distance_kpc' column
            if 'dm' in updated_data.columns:
                updated_data['distance_kpc'] = updated_data['dm'].apply(lambda x: dm_to_kpc(float(x)) if pd.notna(x) else None)

            # Recalculate dm_error in kpc
            if 'dm' in updated_data.columns and 'dm   _error' in updated_data.columns:
                updated_data['distance_kpc_error'] = updated_data.apply(
                    lambda row: np.log(10) * (dm_to_kpc(float(row['dm'])) / 5) * float(row['dm   _error'])
                    if pd.notna(row['dm']) and pd.notna(row['dm   _error']) else None,
                    axis=1
                )


            # Apply calculate_e_bv to the 'Av' column and create a new 'E_BV' column
            if 'Av' in updated_data.columns:
                updated_data['E_BV'] = updated_data['Av'].apply(lambda x: calculate_e_bv(float(x)) if pd.notna(x) else None)

            # Recalculate Av_error in units of E(B-V)
            if 'Av   _error' in updated_data.columns:
                updated_data['E_BV_error'] = updated_data['Av   _error'].apply(
                    lambda x: float(x) / 3.1 if pd.notna(x) else None
                )


            # Save the updated data back to the CSV file
            updated_data.to_csv(csv_file, index=False)

            print(f"Updated CSV file saved: {csv_file}")

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

    except Exception as e:
        loading = False
        print(f"\nError reading or processing the CSV file: {e}")


# Example usage
if __name__ == "__main__":
    csv_file = input("CSV file name:")  # Replace with the path to your CSV file
    process_clusters_from_csv(csv_file)
