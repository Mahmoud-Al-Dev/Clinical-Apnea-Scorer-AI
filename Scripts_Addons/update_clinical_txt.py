import numpy as np
import os

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'OSA'
NIGHT_ID = 1
SAMPLING_RATE = 32 # 32 Hz

# File paths
cleaned_segments_path = f'Y_{TARGET_TYPE}_{NIGHT_ID}_CLEAN.npy'
segment_times_path = f'segment_times_n{NIGHT_ID}.npy'
original_txt_path = r'Data\ON020217-06.TXT' 
output_txt_path = f'ON020217-06_{TARGET_TYPE}_CLEANED.TXT'

def generate_cleaned_clinical_report():
    print(f"--- Generating Updated Clinical Report for Night {NIGHT_ID} ---")
    
    # 1. Load the cleaned numpy arrays
    Y_clean = np.load(cleaned_segments_path)
    segment_times = np.load(segment_times_path)
    
    # 2. Stitch the overlapping segments into a continuous timeline
    min_time = segment_times[0, 0]
    max_time = segment_times[-1, -1]
    total_samples = int(np.ceil((max_time - min_time) * SAMPLING_RATE)) + 1
    
    continuous_labels = np.zeros(total_samples, dtype=int)
    
    for i in range(len(Y_clean)):
        seg_times = segment_times[i]
        seg_labels = Y_clean[i].flatten()
        
        start_idx = int(round((seg_times[0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(seg_labels)
        end_idx = min(end_idx, total_samples)
        
        continuous_labels[start_idx:end_idx] |= seg_labels[:end_idx-start_idx]

    # 3. Extract continuous events
    print("Extracting updated events from the continuous timeline...")
    padded = np.pad(continuous_labels, (1, 1), 'constant')
    diffs = np.diff(padded)
    
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    events = []
    for s, e in zip(starts, ends):
        events.append({
            'start': min_time + (s / SAMPLING_RATE),
            'duration': (e - s) / SAMPLING_RATE
        })

    num_events = len(events)
    print(f"Found {num_events} cleaned {TARGET_TYPE} events.")

    # 4. Smart Parsing to Overwrite Original File
    print(f"Reading original clinical file: {original_txt_path}")
    with open(original_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    in_respiratory_section = False
    wrote_new_events = False

    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        for line in lines:
            
            # STATE 1: Find the header and update the count
            if line.startswith("Respiratory/Apnea/Hypopnea"):
                out_file.write(f"Respiratory/Apnea/Hypopnea ({num_events} Ereignisse)\n")
                in_respiratory_section = True
                continue
            
            # STATE 2: We are in the table header. Copy it, then write OUR events.
            if in_respiratory_section and not wrote_new_events:
                out_file.write(line)
                
                # The dashed line means the table is starting
                if line.startswith("---"):
                    for j, event in enumerate(events):
                        idx = j + 1
                        epoch = int(event['start'] // 30) + 1
                        
                        if j < len(events) - 1:
                            interval = f"{(events[j+1]['start'] - event['start']):.3f},"
                        else:
                            interval = "-,"
                        
                        # Formatting to match the screenshot's columns perfectly
                        idx_str = f"{idx},"
                        epoch_str = f"{epoch},"
                        start_str = f"{event['start']:.3f},"
                        dur_str = f"{event['duration']:.3f},"
                        
                        event_line = f"{idx_str:<4} 1,            {epoch_str:<8} {start_str:<15} {dur_str:<15} {interval:<15} {TARGET_TYPE}\n"
                        out_file.write(event_line)
                    
                    wrote_new_events = True
                continue
            
            # STATE 3: Skip (Delete) the old events
            if in_respiratory_section and wrote_new_events:
                # If we hit an empty line or a new section header, the old table is over.
                if line.strip() == "" or "Sighs" in line or "Body Position" in line:
                    in_respiratory_section = False
                    out_file.write(line) 
                continue 
            
            # STATE 4: Write everything else normally
            if not in_respiratory_section:
                out_file.write(line)

    print(f"\n✅ SUCCESS! Old events successfully overwritten. Saved to: {output_txt_path}")

if __name__ == "__main__":
    generate_cleaned_clinical_report()