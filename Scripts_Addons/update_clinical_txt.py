import numpy as np
import os

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'CA'
NIGHT_ID = 3
SAMPLING_RATE = 32 # 32 Hz

cleaned_segments_path = f'Y_{TARGET_TYPE}_{NIGHT_ID}_SILVER.npy'
original_segments_path = f'Y_{TARGET_TYPE}_{NIGHT_ID}.npy'
segment_times_path = f'segment_times_n{NIGHT_ID}.npy'

original_txt_path = r'Data\ST300317-06.TXT' 
output_txt_path = f'ST300317-06_{TARGET_TYPE}_CLEANED.TXT'

def generate_cleaned_clinical_report():
    print(f"--- Chronologically Integrating AI Discoveries for Night {NIGHT_ID} ---")
    
    Y_clean = np.load(cleaned_segments_path)
    Y_orig = np.load(original_segments_path)
    segment_times = np.load(segment_times_path)
    
    # Isolate ONLY the new events the AI added.
    Y_ai_only = np.clip(Y_clean - Y_orig, 0, 1)
    
    min_time = segment_times[0, 0]
    max_time = segment_times[-1, -1]
    total_samples = int(np.ceil((max_time - min_time) * SAMPLING_RATE)) + 1
    
    continuous_labels = np.zeros(total_samples, dtype=int)
    
    for i in range(len(Y_ai_only)):
        seg_times = segment_times[i]
        seg_labels = Y_ai_only[i].flatten()
        
        start_idx = int(round((seg_times[0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(seg_labels)
        end_idx = min(end_idx, total_samples)
        
        continuous_labels[start_idx:end_idx] |= seg_labels[:end_idx-start_idx]

    padded = np.pad(continuous_labels, (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    # Store AI events as dictionaries
    ai_events = []
    for s, e in zip(starts, ends):
        ai_events.append({
            'start': min_time + (s / SAMPLING_RATE),
            'duration': (e - s) / SAMPLING_RATE,
            'type': f'{TARGET_TYPE}' # Tag them so they stand out!
        })

    print(f"Found {len(ai_events)} BRAND NEW AI events to weave into the timeline.")

    # Parse the original file
    with open(original_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    out_lines = []
    state = 'PRE_TABLE'
    header_line_idx = -1
    parsed_original_events = []

    for line in lines:
        if state == 'PRE_TABLE':
            if line.startswith("Respiratory/Apnea/Hypopnea"):
                state = 'IN_HEADER'
                header_line_idx = len(out_lines)
                out_lines.append(line) # Placeholder, we will update the count later
            else:
                out_lines.append(line)

        elif state == 'IN_HEADER':
            out_lines.append(line)
            if line.startswith("---"):
                state = 'IN_EVENTS'

        elif state == 'IN_EVENTS':
            # Stop condition for the table
            if line.strip() == "" or "Sighs" in line or "Body Position" in line:
                
                # --- THE MAGIC CHRONOLOGICAL WEAVE ---
                all_events = parsed_original_events + ai_events
                all_events.sort(key=lambda x: x['start']) # Sort everything by start time!
                
                # Update the header count
                out_lines[header_line_idx] = f"Respiratory/Apnea/Hypopnea ({len(all_events)} Ereignisse)\n"
                
                # Format and write the sorted timeline
                for j, ev in enumerate(all_events):
                    idx = j + 1
                    epoche = int(ev['start'] // 30) + 1
                    
                    # Recalculate the interval to the NEXT chronological event
                    if j < len(all_events) - 1:
                        interval = f"{(all_events[j+1]['start'] - ev['start']):.3f}"
                    else:
                        interval = "-"
                        
                    # String formatting to match the clinical layout
                    idx_str = f"{idx},"
                    abs_str = "1," 
                    ep_str = f"{epoche},"
                    st_str = f"{ev['start']:.3f},"
                    dur_str = f"{ev['duration']:.3f},"
                    int_str = f"{interval}," if interval != "-" else "-,"
                    
                    event_line = f"{idx_str:<4} {abs_str:<3} {ep_str:<5} {st_str:<12} {dur_str:<10} {int_str:<12} {ev['type']}\n"
                    out_lines.append(event_line)

                # Table is finished, append the line that broke the loop and switch state
                out_lines.append(line)
                state = 'POST_TABLE'
            else:
                # Read the doctor's original events and save them to memory
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    try:
                        parsed_original_events.append({
                            'start': float(parts[3]),
                            'duration': float(parts[4]),
                            'type': parts[6]
                        })
                    except ValueError:
                        pass # Skip malformed lines if any

        elif state == 'POST_TABLE':
            out_lines.append(line)

    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        out_file.writelines(out_lines)

    print(f"\n✅ SUCCESS! {len(ai_events)} AI events woven chronologically into the original labels. Saved to: {output_txt_path}")

if __name__ == "__main__":
    generate_cleaned_clinical_report()