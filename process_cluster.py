import subprocess
import pandas as pd

def process_cluster(cluster_name):
    try:
        # Run isofitting.py with the cluster name as an argument and capture its output
        result = subprocess.run(
            ["python", "isofitting.py", cluster_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully processed cluster: {cluster_name}")
        # Parse the output from isofitting.py
        output_lines = result.stdout.strip().split("\n")
        parameters = {}
        for line in output_lines:
            if ":" in line:  # Assuming output is in "key: value" format
                key, value = line.split(":", 1)
                parameters[key.strip()] = value.strip()
        return parameters
    except subprocess.CalledProcessError as e:
        print(f"Error processing cluster {cluster_name}: {e}")
        return None

def process_clusters_from_csv(csv_file):
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)
        # Ensure the column containing cluster names is named 'cluster_name'
        if 'cluster_name' not in data.columns:
            print("Error: CSV file must contain a 'cluster_name' column.")
            return
        # Create a list to store the results
        results = []
        # Iterate over each cluster name and process it
        for cluster_name in data['cluster_name']:
            parameters = process_cluster(cluster_name)
            if parameters:
                parameters['cluster_name'] = cluster_name
                results.append(parameters)
        # Convert results to a DataFrame and merge with the original data
        if results:
            results_df = pd.DataFrame(results)
            updated_data = pd.merge(data, results_df, on='cluster_name', how='left')
            # Save the updated data back to the CSV file
            updated_data.to_csv(csv_file, index=False)
            print(f"Updated CSV file saved: {csv_file}")
    except Exception as e:
        print(f"Error reading or processing the CSV file: {e}")

# Example usage
if __name__ == "__main__":
    csv_file = "trial1.csv"  # Replace with the path to your CSV file
    process_clusters_from_csv(csv_file)