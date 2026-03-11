import numpy as np
import os

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TXT_FILE_PATH = 'Data\ON030217-06.TXT'  
SEGMENT_TIMES_PATH = 'segment_times.npy'
# ==========================================

print("1. Loading saved segment timeline...")
if not os.path.exists(SEGMENT_TIMES_PATH):
    print(f"❌ Error: {SEGMENT_TIMES_PATH} not found. Run the signal processing script first!")
    exit()

# segment_times shape: (Num_Segments, 960)
segment_times = np.load(SEGMENT_TIMES_PATH)
num_segments, num_timesteps = segment_times.shape

print("2. Parsing doctor's clinical notes...")
events = []

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
        if not line.strip() and len(events) > 0:
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
                
                # Map the text to your AI classes
                if event_str == 'CA':
                    event_class = 1
                elif event_str == 'OH' or event_str == 'OA': # Added OA (Obstructive Apnea) just in case!
                    event_class = 2
                else:
                    continue # Skip anything else we don't care about right now
                
                events.append((start_t, end_t, event_class))
                
            except ValueError:
                # This catches any weird lines that don't have numbers where numbers should be
                continue

print(f"   Found {len(events)} valid clinical events (CA/OH).")

print("3. Syncing clinical events to AI time segments...")
# Initialize the target tensor with zeros (Class 0 = Normal Breathing)
Y_labels = np.zeros((num_segments, num_timesteps, 1), dtype=np.int32)

# Loop through every 30-second AI window
for i in range(num_segments):
    # This array holds the 960 exact timestamps for this specific window
    window_time_axis = segment_times[i] 
    
    # Check every doctor event
    for start_t, end_t, event_class in events:
        
        # Find exactly which indices in this window fall inside the doctor's start and end times.
        # This completely ignores "stitched gaps" because it looks at the absolute timestamp!
        overlap_mask = (window_time_axis >= start_t) & (window_time_axis <= end_t)
        
        # Assign the class (1 or 2) to those specific pixels
        if np.any(overlap_mask):
            Y_labels[i, overlap_mask, 0] = event_class

print("4. Saving Y_train_Labels.npy...")
np.save('Y_train_Labels.npy', Y_labels)

# Print a quick summary to prove it worked
total_ca = np.sum(Y_labels == 1)
total_oh = np.sum(Y_labels == 2)
print(f"✅ SUCCESS!")
print(f"   Y_train shape: {Y_labels.shape}")
print(f"   Total CA (Class 1) data points assigned: {total_ca}")
print(f"   Total OH (Class 2) data points assigned: {total_oh}")