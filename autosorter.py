import os
import shutil
import csv

def sort_all_physionet_data():
    # 1. Define absolute paths based on your system
    SOURCE_BASE = "/Users/neel/Downloads/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
    PROJECT_BASE = "/Users/neel/Downloads/avf_stenosis"
    
    DEST_HEALTHY = os.path.join(PROJECT_BASE, "data", "healthy")
    DEST_STENOTIC = os.path.join(PROJECT_BASE, "data", "stenotic")
    
    # 2. Create the destination folders in your project directory
    os.makedirs(DEST_HEALTHY, exist_ok=True)
    os.makedirs(DEST_STENOTIC, exist_ok=True)
    
    # 3. List the exact folders from the PhysioNet dataset
    training_folders = [
        "training-a", "training-b", "training-c", 
        "training-d", "training-e", "training-f"
    ]
    
    total_healthy = 0
    total_stenotic = 0
    
    # 4. Loop through every folder using absolute paths
    for folder_name in training_folders:
        folder_path = os.path.join(SOURCE_BASE, folder_name)
        csv_path = os.path.join(folder_path, "REFERENCE.csv")
        
        if not os.path.exists(csv_path):
            print(f"Skipping {folder_name}: No REFERENCE.csv found at {csv_path}")
            continue
            
        print(f"Scanning and extracting files from {folder_name}...")
        
        with open(csv_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                # Some CSVs have the .wav extension, some don't. We handle both.
                filename = row[0]
                if not filename.endswith('.wav'):
                    filename += ".wav"
                    
                label = row[1]
                source_file = os.path.join(folder_path, filename)
                
                # Copy the file if it exists
                if os.path.exists(source_file):
                    if label == "1":
                        # Label 1 = Normal (Healthy)
                        target_path = os.path.join(DEST_HEALTHY, filename)
                        shutil.copy2(source_file, target_path)
                        total_healthy += 1
                        
                    elif label == "-1":
                        # Label -1 = Abnormal (Stenotic/Murmur)
                        target_path = os.path.join(DEST_STENOTIC, filename)
                        shutil.copy2(source_file, target_path)
                        total_stenotic += 1

    # 5. Print the final results
    print("\n========================================")
    print("✅ Sorting Complete!")
    print(f"Total Healthy files copied: {total_healthy}")
    print(f"Total Stenotic files copied: {total_stenotic}")
    print(f"Data saved to: {os.path.join(PROJECT_BASE, 'data')}")
    print("========================================")

if __name__ == "__main__":
    sort_all_physionet_data()