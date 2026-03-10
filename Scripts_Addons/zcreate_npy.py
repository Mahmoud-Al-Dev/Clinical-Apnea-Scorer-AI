import numpy as np

# --- CONFIGURATION (Must match your existing pipeline) ---
DATA_START_TIME = 10000   # Start of your current X_train data
WINDOW_SEC = 30           # 30-second windows
STEP_SEC = 20             # 20-second step
FS = 32                   # 32 Hz sampling rate
NUM_SEGMENTS = 49         # Total windows in your 1000s block

# --- CLINICAL DATA FROM YOUR IMAGE ---
# Format: (Start Time in seconds, Duration in seconds)
doctor_events = [
    (10490.349, 21.355),
    (10549.281, 12.115),
    (10663.655, 14.168),
    (10878.439, 14.990),
    (10923.819, 15.811)
]

def create_clinical_labels():
    print(f"Extracting clinical labels for {NUM_SEGMENTS} segments...")
    
    # Initialize the empty labels array (Segments, 960 samples, 1)
    Y_labels = np.zeros((NUM_SEGMENTS, 960, 1), dtype=np.int32)
    
    for i in range(NUM_SEGMENTS):
        # Calculate the real-world time boundaries for this specific window
        window_start = DATA_START_TIME + (i * STEP_SEC)
        window_end = window_start + WINDOW_SEC
        
        for event_start, duration in doctor_events:
            event_end = event_start + duration
            
            # Check if this doctor's event overlaps with our current window
            overlap_start = max(window_start, event_start)
            overlap_end = min(window_end, event_end)
            
            if overlap_start < overlap_end:
                # Convert the overlapping time into local 0-960 sample indices
                idx_start = int((overlap_start - window_start) * FS)
                idx_end = int((overlap_end - window_start) * FS)
                
                # Fill the binary mask for this window
                Y_labels[i, idx_start:idx_end, 0] = 1
                
    # Save as the new ground truth
    np.save('Y_train_Labels.npy', Y_labels)
    print(f"✅ Successfully saved clinical Y_train_Labels.npy")

if __name__ == "__main__":
    create_clinical_labels()