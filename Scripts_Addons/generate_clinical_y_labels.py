import numpy as np
import os

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TXT_FILE_PATH = 'Data/9MG080916-07.TXT'  
TEST_NIGHT=9
SEGMENT_TIMES_PATH = f'Nights/segment_times_n{TEST_NIGHT}.npy'
# ==========================================

print("1. Loading saved segment timeline...")
if not os.path.exists(SEGMENT_TIMES_PATH):
    print(f"❌ Error: {SEGMENT_TIMES_PATH} not found. Run the signal processing script first!")
    exit()

# segment_times shape: (Num_Segments, 960)
segment_times = np.load(SEGMENT_TIMES_PATH)
num_segments, num_timesteps = segment_times.shape

print("2. Parsing doctor's clinical notes...")
events_ca = []
events_osa = []

# Open and parse the text file
with open(TXT_FILE_PATH, 'r', encoding='latin-1') as file:
    lines = file.readlines()

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
for i in range(num_segments):
    window_time_axis = segment_times[i] 
    
    # --- Apply CA Targets ---
    for start_t, end_t in events_ca:
        overlap_mask = (window_time_axis >= start_t) & (window_time_axis <= end_t)
        if np.any(overlap_mask):
            Y_labels_CA[i, overlap_mask, 0] = 1 # Mark 1 for CA

    # --- Apply OSA Targets ---
    for start_t, end_t in events_osa:
        overlap_mask = (window_time_axis >= start_t) & (window_time_axis <= end_t)
        if np.any(overlap_mask):
            Y_labels_OSA[i, overlap_mask, 0] = 1 # Mark 1 for OSA

print("4. Saving Y_train_Labels_CA.npy and Y_train_Labels_OSA.npy...")
np.save(f'Y_CA_{TEST_NIGHT}.npy', Y_labels_CA)
np.save(f'Y_OSA_{TEST_NIGHT}.npy', Y_labels_OSA)

# Print a quick summary to prove it worked
print(f"✅ SUCCESS!")
print(f"   CA Array shape:  {Y_labels_CA.shape} | Total CA data points:  {np.sum(Y_labels_CA == 1)}")
print(f"   OSA Array shape: {Y_labels_OSA.shape} | Total OSA data points: {np.sum(Y_labels_OSA == 1)}")