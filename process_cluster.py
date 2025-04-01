import subprocess
import pandas as pd
import threading
import time

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
        print(f"\nSuccessfully processed cluster: {cluster_name}")
        
        # Parse the output from isofitting.py
        output_lines = result.stdout.strip().split("\n")
        parameters = {}
        for line in output_lines:
            # Capture only lines matching the format "key: value" or "key_error: value"
            if any(param in line for param in ["met", "loga", "dm", "Av", "met_error", "loga_error", "dm_error", "Av_error"]):
                key, value = line.split(":", 1)
                parameters[key.strip()] = value.strip()

                # Ensure only the relevant columns are returned
        relevant_keys = ["met", "met  _error", "loga", "loga _error", "dm", "dm   _error", "Av", "Av   _error"]
        filtered_parameters = {key: parameters.get(key, None) for key in relevant_keys}
        return filtered_parameters

    except subprocess.CalledProcessError as e:
        print(f"\nError processing cluster {cluster_name}: {e.stderr}")
        return None

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
    csv_file = "trial2.csv"  # Replace with the path to your CSV file
    process_clusters_from_csv(csv_file)