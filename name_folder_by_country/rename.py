from pathlib import Path
import json

# Define the base directory where the numbered folders are located
base_dir = Path('dataset_renametest')  # Replace with your actual path

# Function to process each folder and extract information
def process_folders(base_dir):
    # Iterate through each folder in the base directory
    for folder in base_dir.iterdir():
        # Check if the current item is a directory and a number
        if folder.is_dir() and folder.name.isdigit():
            # Find the STAC JSON file inside the numbered folder
            stac_json_path = folder / f"sentinel12_{folder.name}_meta.json"

            if stac_json_path.exists():
                # Open and load the JSON file
                with stac_json_path.open('r') as f:
                    stac_data = json.load(f)

                # Extract the required metadata (date and country)
                date = stac_data['properties'].get('date_s1', 'unknown_date')
                landcover = stac_data['properties'].get('landcover', 'unknown_landcover')

                # You could extract country from another part of the metadata if available,
                # or infer it from landcover, but here's an example using landcover.
                # If "country" were available, you would use it like:
                # country = stac_data['properties'].get('country', 'unknown_country')
                
                # Print or process the extracted information
                print(f"Folder: {folder.name}, Date: {date}, Landcover: {landcover}")
                
                # You could further process or rename the folder using this information
                # For example, renaming the folder based on date and landcover:
                new_folder_name = f"{folder.name}_{landcover.replace(' ', '_').lower()}_{date}"
                new_folder_path = folder.with_name(new_folder_name)
                folder.rename(new_folder_path)
                print(f"Renamed folder to: {new_folder_path}")

# Run the function to process folders
process_folders(base_dir)
