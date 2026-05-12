import os
import json

def concatenate_sessions(raw_folder, output_file):
    all_sessions = []
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Iterate through all files in the raw directory
    for filename in os.listdir(raw_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(raw_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # If the file contains a list, extend our main list
                    if isinstance(data, list):
                        all_sessions.extend(data)
                    # If it's a single object, append it
                    else:
                        all_sessions.append(data)
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {filename}")

    # Write the concatenated result to the interim folder
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_sessions, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully concatenated files into: {output_file}")
    print(f"Total sessions processed: {len(all_sessions)}")

# Define paths
raw_data_path = 'data/raw'
interim_data_path = 'data/interim/all_sessions.json'

# Run the concatenation
concatenate_sessions(raw_data_path, interim_data_path)