import pandas as pd
import numpy as np
import os

# ==========================================
# --- USER CONTROLS ---
# ==========================================
INPUT_FILE = 'Data/3ON020217-06.csv'  # The single large night file
OUTPUT_FOLDER = 'Data_Split'          # Folder to save the 5 chunks
NUM_SPLITS = 5
# ==========================================

def split_csv_night(file_path, num_splits, output_dir):
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base filename without the '.csv' extension (e.g., 'ZED270417-05')
    base_name = os.path.basename(file_path).replace('.csv', '')

    print(f" Loading {file_path}...")
    try:
        # Read the CSV completely raw (header=None prevents it from eating the first row)
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f" Error: File {file_path} not found.")
        return

    total_rows = len(df)
    print(f"   Total rows loaded: {total_rows}")

    # 2. Split the DataFrame into exactly 5 chunks
    # np.array_split safely handles remainders if total_rows isn't perfectly divisible by 5
    chunks = np.array_split(df, num_splits)

    print(f" Splitting into {num_splits} equal files...")
    
    # 3. Save each chunk
    for i, chunk in enumerate(chunks):
        part_number = i + 1 # 1-indexed (Part 1 to 5)
        output_file = os.path.join(output_dir, f"{base_name}_part{part_number}.csv")
        
        # Save without index and without header to maintain your exact raw format
        chunk.to_csv(output_file, index=False, header=False)
        print(f"   ✅ Saved: {output_file} ({len(chunk)} rows)")

    print(f"\n Split complete! All files saved in '{output_dir}/'")

if __name__ == "__main__":
    split_csv_night(INPUT_FILE, NUM_SPLITS, OUTPUT_FOLDER)