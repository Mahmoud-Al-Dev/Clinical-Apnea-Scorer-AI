import numpy as np
import os

# ==========================================
# --- USER CONTROLS ---
# ==========================================
# List of night IDs you want to process
TEST_NIGHTS = [26, 27, 28]  

# Corresponding list of clinical text files. 
# Index 0 matches TEST_NIGHTS[0], Index 1 matches TEST_NIGHTS[1], etc.
TXT_FILE_PATHS = [
    'Data/AB100417-05.TXT',  # Corresponds to Night 26
    'Data/AD221116-05.TXT',  # Corresponds to Night 27
    'Data/AL150117-10.TXT'   # Corresponds to Night 28
]
# ==========================================

# 0. Safety Check
if len(TEST_NIGHTS) != len(TXT_FILE_PATHS):
    print("❌ Error: The number of TEST_NIGHTS must exactly match the number of TXT_FILE_PATHS.")
    exit()

print(f"--- Starting batch processing for {len(TEST_NIGHTS)} nights ---")

for i, night_id in enumerate(TEST_NIGHTS):
    txt_path = TXT_FILE_PATHS[i]
    segment_times_path = f'Nights/segment_times_n{night_id}.npy'
    
    print(f"\n==========================================")
    print(f"▶ Processing Night {night_id} using {txt_path}")
    print(f"==========================================")

    print("1. Loading saved segment timeline...")
    if not os.path.exists(segment_times_path):
        print(f"⚠️ Warning: {segment_times_path} not found. Skipping Night {night_id}...")
        continue

    # segment_times shape: (Num_Segments, 960)
    segment_times = np.load(segment_times_path)
    num_segments, num_timesteps = segment_times.shape

    print("2. Parsing doctor's clinical notes...")
    events_ca = []
    events_osa = []

    # Open and parse the text file
    try:
        with open(txt_path, 'r', encoding='latin-1') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"⚠️ Warning: Could not find {txt_path}. Skipping Night {night_id}...")
        continue

    in_respiratory_section = False

    for line in lines:
        # Look for the start of the table
        if "Respiratory/Apnea/Hypopnea" in line:
            in_respiratory_section = True
            continue
        
        # If we are inside the table, parse the data
        if in_respiratory_section:
            # Stop parsing if we hit an empty line or a new section
            if not line.strip() and (len(events_ca) > 0 or len(events_osa) > 0):
                break
                
            # Skip the dashed border lines or header rows
            if line.strip().startswith('-') or line.strip().startswith('#'):
                continue
                
            # Split the line by commas
            parts = line.split(',')
            
            # Make sure the line actually has the 7 columns we expect
            if len(parts) >= 7:
                try:
                    # Column 3 is Startzeit, Column 4 is Dauer, Column 6 is Ereignis
                    start_t = float(parts[3].strip())
                    duration = float(parts[4].strip())
                    end_t = start_t + duration
                    
                    event_str = parts[6].strip()
                    
                    # Route the event to the correct list
                    if event_str == 'CA':
                        events_ca.append((start_t, end_t))
                    elif event_str in ['OH', 'OA', 'OSA']: # Catching both Obstructive Hypopnea and Apnea
                        events_osa.append((start_t, end_t))
                    else:
                        continue # Skip everything else
                    
                except ValueError:
                    # This catches any weird lines that don't have numbers where numbers should be
                    continue

    print(f"   Found {len(events_ca)} CA events and {len(events_osa)} OSA events.")

    print("3. Syncing clinical events to AI time segments...")
    # Initialize TWO target tensors with zeros (Class 0 = Normal Breathing)
    Y_labels_CA = np.zeros((num_segments, num_timesteps, 1), dtype=np.int32)
    Y_labels_OSA = np.zeros((num_segments, num_timesteps, 1), dtype=np.int32)

    # Loop through every 30-second AI window
    for j in range(num_segments):
        window_time_axis = segment_times[j] 
        
        # --- Apply CA Targets ---
        for start_t, end_t in events_ca:
            overlap_mask = (window_time_axis >= start_t) & (window_time_axis <= end_t)
            if np.any(overlap_mask):
                Y_labels_CA[j, overlap_mask, 0] = 1 # Mark 1 for CA

        # --- Apply OSA Targets ---
        for start_t, end_t in events_osa:
            overlap_mask = (window_time_axis >= start_t) & (window_time_axis <= end_t)
            if np.any(overlap_mask):
                Y_labels_OSA[j, overlap_mask, 0] = 1 # Mark 1 for OSA

    ca_save_path = f'Nights/Y_CA_{night_id}.npy'
    osa_save_path = f'Nights/Y_OSA_{night_id}.npy'

    print(f"4. Saving arrays to Nights folder...")
    np.save(ca_save_path, Y_labels_CA)
    np.save(osa_save_path, Y_labels_OSA)

    # Print a quick summary to prove it worked
    print(f"✅ Night {night_id} SUCCESS!")
    print(f"   CA Array shape:  {Y_labels_CA.shape} | Total CA data points:  {np.sum(Y_labels_CA == 1)}")
    print(f"   OSA Array shape: {Y_labels_OSA.shape} | Total OSA data points: {np.sum(Y_labels_OSA == 1)}")

print("\n🎉 Batch processing complete for all assigned nights!")