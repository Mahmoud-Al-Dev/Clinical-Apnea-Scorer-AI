import numpy as np
import os

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPES = ['OSA', 'CA'] 
NIGHT_ID = 8
SAMPLING_RATE = 32 # 32 Hz

segment_times_path = f'Nights/segment_times_n{NIGHT_ID}.npy'
original_txt_path = r'Data/UI220916-06.TXT' 
output_txt_path = f'UI220916-06_MASTER_SILVER_CLEANED.TXT'
# ==========================================

def process_target_data(target_type, segment_times, min_time, total_samples):
    cleaned_path = f'Nights/Y_{target_type}_{NIGHT_ID}_SILVER.npy'
    orig_path = f'Nights/Y_{target_type}_{NIGHT_ID}.npy'
    
    if not os.path.exists(cleaned_path) or not os.path.exists(orig_path):
        print(f"  ⚠️ Warning: Missing SILVER or Original files for {target_type}. Skipping this type.")
        return [], None

    print(f"  -> Processing {target_type} data...")
    Y_clean = np.load(cleaned_path)
    Y_orig = np.load(orig_path)
    
    # 1. Build continuous timeline of SILVER standard (to verify doctor labels)
    continuous_clean = np.zeros(total_samples, dtype=int)
    for i in range(len(Y_clean)):
        seg_times = segment_times[i]
        seg_labels = Y_clean[i].flatten()
        start_idx = int(round((seg_times[0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(seg_labels)
        end_idx = min(end_idx, total_samples)
        continuous_clean[start_idx:end_idx] |= seg_labels[:end_idx-start_idx]

    # 2. Isolate ONLY the new events the AI added
    Y_ai_only = np.clip(Y_clean - Y_orig, 0, 1)
    continuous_ai_added = np.zeros(total_samples, dtype=int)
    
    for i in range(len(Y_ai_only)):
        seg_times = segment_times[i]
        seg_labels = Y_ai_only[i].flatten()
        start_idx = int(round((seg_times[0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(seg_labels)
        end_idx = min(end_idx, total_samples)
        continuous_ai_added[start_idx:end_idx] |= seg_labels[:end_idx-start_idx]

    padded = np.pad(continuous_ai_added, (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    ai_events = []
    for s, e in zip(starts, ends):
        ai_events.append({
            'start': min_time + (s / SAMPLING_RATE),
            'duration': (e - s) / SAMPLING_RATE,
            'type': target_type
        })
        
    print(f"     Found {len(ai_events)} new AI discoveries for {target_type}.")
    return ai_events, continuous_clean

def generate_master_clinical_report():
    print(f"--- Chronologically Integrating AI Discoveries for Night {NIGHT_ID} ---")
    
    segment_times = np.load(segment_times_path)
    min_time = segment_times[0, 0]
    max_time = segment_times[-1, -1]
    total_samples = int(np.ceil((max_time - min_time) * SAMPLING_RATE)) + 1
    
    all_ai_events = []
    clean_timelines = {}
    
    # Process both OSA and CA arrays
    for t_type in TARGET_TYPES:
        events, timeline = process_target_data(t_type, segment_times, min_time, total_samples)
        if timeline is not None:
            all_ai_events.extend(events)
            clean_timelines[t_type] = timeline

    print(f"\nTotal BRAND NEW AI events to weave into the timeline: {len(all_ai_events)}")

    # Parse the original file and FILTER OUT deleted doctor events
    with open(original_txt_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    out_lines = []
    state = 'PRE_TABLE'
    header_line_idx = -1
    parsed_original_events = []
    deleted_doctor_events = {t_type: 0 for t_type in TARGET_TYPES}

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
            if line.strip() == "" or "Sighs" in line or "Body Position" in line:
                
                # --- THE MAGIC CHRONOLOGICAL WEAVE ---
                all_events = parsed_original_events + all_ai_events
                all_events.sort(key=lambda x: x['start']) # Sort everything chronologically!
                
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

                out_lines.append(line)
                state = 'POST_TABLE'
            else:
                # Read the doctor's original events
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    try:
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        event_type = parts[6]
                        
                        keep_event = True
                        
                        # --- VERIFY DOCTOR'S LABEL SURVIVED YOUR REVIEW ---
                        if event_type in clean_timelines:
                            start_idx = int(round((start_time - min_time) * SAMPLING_RATE))
                            end_idx = int(round((start_time + duration - min_time) * SAMPLING_RATE))
                            timeline = clean_timelines[event_type]
                            
                            if 0 <= start_idx < len(timeline):
                                end_idx = min(end_idx, len(timeline))
                                # If there are ZERO '1s' in this window, you deleted it!
                                if np.sum(timeline[start_idx:end_idx]) == 0:
                                    keep_event = False
                                    deleted_doctor_events[event_type] += 1
                        
                        if keep_event:
                            parsed_original_events.append({
                                'start': start_time,
                                'duration': duration,
                                'type': event_type
                            })
                    except ValueError:
                        pass 

        elif state == 'POST_TABLE':
            out_lines.append(line)

    with open(output_txt_path, 'w', encoding='latin-1') as out_file:
        out_file.writelines(out_lines)

    print("\n--- Summary of Deleted Doctor False Alarms ---")
    for t_type, count in deleted_doctor_events.items():
        print(f"Purged {count} incorrect {t_type} labels.")
        
    print(f"\n✅ SUCCESS! Master timeline saved to: {output_txt_path}")

if __name__ == "__main__":
    generate_master_clinical_report()