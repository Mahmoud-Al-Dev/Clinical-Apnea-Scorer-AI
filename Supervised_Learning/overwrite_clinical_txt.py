import numpy as np
import os

# ==========================================
# --- CONFIGURATION ---
# ==========================================
NIGHT_ID = 6
SAMPLING_RATE = 32 # 32 Hz

osa_segments_path = f'Nights/Y_OSA_{NIGHT_ID}_ADJUSTED.npy'
ca_segments_path = f'Nights/Y_CA_{NIGHT_ID}_ADJUSTED.npy'
segment_times_path = f'Nights/segment_times_n{NIGHT_ID}.npy'

original_txt_path = r'Data\ST310317-06.TXT' 
output_txt_path = f'ST310317-06_MASTER_CLEANED.TXT'
# ==========================================

def extract_events_from_array(npy_path, target_type, segment_times, min_time, total_samples):
    if not os.path.exists(npy_path):
        print(f"  ⚠️ Warning: {npy_path} not found. Skipping {target_type} events.")
        return []
        
    print(f"  -> Extracting {target_type} events from {npy_path}...")
    Y_data = np.load(npy_path)
    
    # Build continuous timeline to naturally merge any overlaps
    continuous_labels = np.zeros(total_samples, dtype=int)
    for i in range(len(Y_data)):
        seg_times = segment_times[i]
        seg_labels = Y_data[i].flatten()
        start_idx = int(round((seg_times[0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(seg_labels)
        end_idx = min(end_idx, total_samples)
        continuous_labels[start_idx:end_idx] |= seg_labels[:end_idx-start_idx]
        
    padded = np.pad(continuous_labels, (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    events = []
    for s, e in zip(starts, ends):
        events.append({
            'start': min_time + (s / SAMPLING_RATE),
            'duration': (e - s) / SAMPLING_RATE,
            'type': target_type
        })
        
    print(f"     Found {len(events)} {target_type} events.")
    return events

def generate_master_clinical_report():
    print(f"--- Generating MASTER Overwritten Report for Night {NIGHT_ID} ---")
    
    if not os.path.exists(segment_times_path):
        print(f"❌ Error: Cannot find {segment_times_path}")
        return
        
    segment_times = np.load(segment_times_path)
    min_time = segment_times[0, 0]
    max_time = segment_times[-1, -1]
    total_samples = int(np.ceil((max_time - min_time) * SAMPLING_RATE)) + 1
    
    # 1. Extract and combine all events
    osa_events = extract_events_from_array(osa_segments_path, 'OSA', segment_times, min_time, total_samples)
    ca_events = extract_events_from_array(ca_segments_path, 'CA', segment_times, min_time, total_samples)
    
    all_events = osa_events + ca_events
    
    # Sort everything chronologically!
    all_events.sort(key=lambda x: x['start']) 
    print(f"\nTotal verified events to inject: {len(all_events)}")

    # 2. Parse the original text file
    with open(original_txt_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    out_lines = []
    state = 'PRE_TABLE'
    header_line_idx = -1

    for line in lines:
        if state == 'PRE_TABLE':
            if line.startswith("Respiratory/Apnea/Hypopnea"):
                state = 'IN_HEADER'
                header_line_idx = len(out_lines)
                out_lines.append(line) 
            else:
                out_lines.append(line)

        elif state == 'IN_HEADER':
            out_lines.append(line)
            if line.startswith("---"):
                state = 'IN_EVENTS'

        elif state == 'IN_EVENTS':
            # We skip EVERYTHING in the doctor's table until the table ends.
            # Tables usually end with a blank line, or a new section like "Sighs" or "Body Position"
            if line.strip() == "" or "Sighs" in line or "Body Position" in line:
                
                # --- INJECT OUR PERFECT TIMELINE ---
                out_lines[header_line_idx] = f"Respiratory/Apnea/Hypopnea ({len(all_events)} Ereignisse)\n"
                
                for j, ev in enumerate(all_events):
                    idx = j + 1
                    epoche = int(ev['start'] // 30) + 1
                    
                    if j < len(all_events) - 1:
                        interval = f"{(all_events[j+1]['start'] - ev['start']):.3f}"
                    else:
                        interval = "-"
                        
                    idx_str = f"{idx},"
                    abs_str = "1," 
                    ep_str = f"{epoche},"
                    st_str = f"{ev['start']:.3f},"
                    dur_str = f"{ev['duration']:.3f},"
                    int_str = f"{interval}," if interval != "-" else "-,"
                    
                    event_line = f"{idx_str:<4} {abs_str:<3} {ep_str:<5} {st_str:<12} {dur_str:<10} {int_str:<12} {ev['type']}\n"
                    out_lines.append(event_line)

                # Append the line that broke us out of the table (e.g. blank line or next section)
                out_lines.append(line)
                state = 'POST_TABLE'
            else:
                # WE DO NOTHING HERE. This effectively deletes the doctor's original labels.
                pass 

        elif state == 'POST_TABLE':
            out_lines.append(line)

    with open(output_txt_path, 'w', encoding='latin-1') as out_file:
        out_file.writelines(out_lines)

    print(f"\n✅ SUCCESS! Master timeline saved to: {output_txt_path}")

if __name__ == "__main__":
    generate_master_clinical_report()