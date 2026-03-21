import os

# ==========================================
# --- USER CONTROLS ---
# ==========================================
INPUT_TXT_PATH = r'Data\ON030217-06.TXT'
OUTPUT_TXT_PATH = r'Data\ON030217-06_RLHF_Updated.TXT'

# Define your new AI-discovered events here!
# Format: (Start_Time, End_Time, Event_Type)
NEW_EVENTS = [
    (6301.0, 6315.0, 'CA'),  
    (6713.26, 6730.0, 'CA'),
    (8241.0, 8260.0, 'CA'),
    (8408.64, 8428.0, 'CA'),
    (9000.0, 9019.0, 'CA'),
    (9300.0, 9316.0, 'CA'),
    (9662.0, 9676.0, 'CA'),
    (10618.0, 10633.0, 'CA')
]
# ==========================================

def insert_events_into_txt():
    if not os.path.exists(INPUT_TXT_PATH):
        print(f"❌ Error: Could not find {INPUT_TXT_PATH}")
        return

    with open(INPUT_TXT_PATH, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    out_lines = []
    events = []
    in_respiratory_section = False
    table_start_idx = -1
    table_end_idx = -1

    print("1. Parsing existing clinical events...")
    
    # First Pass: Extract all existing events
    for i, line in enumerate(lines):
        if "Respiratory/Apnea/Hypopnea" in line:
            in_respiratory_section = True
            table_start_idx = i
            continue
            
        if in_respiratory_section:
            # Stop if we hit an empty line after the table starts
            if not line.strip() and len(events) > 0:
                table_end_idx = i
                in_respiratory_section = False
                continue
                
            # Skip the dashed lines or headers
            if line.strip().startswith('-') or line.strip().startswith('#'):
                continue
                
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    start_t = float(parts[3].strip())
                    duration = float(parts[4].strip())
                    event_type = parts[6].strip()
                    events.append({
                        'start': start_t,
                        'duration': duration,
                        'type': event_type
                    })
                except ValueError:
                    continue

    print(f"   Found {len(events)} existing events.")

    # Add the NEW events into the list
    for start_t, end_t, e_type in NEW_EVENTS:
        events.append({
            'start': start_t,
            'duration': end_t - start_t,
            'type': e_type
        })
        print(f"   Added NEW event: {e_type} at {start_t}s")

    # Sort ALL events strictly by Start Time
    events = sorted(events, key=lambda x: x['start'])

    print("2. Recalculating Epochs and Intervals...")
    # Calculate Intervals, Epochs, and format the strings
    formatted_event_lines = []
    for i, ev in enumerate(events):
        idx = i + 1
        abschnitt = 1 # Default
        epoch = int(ev['start'] / 30) + 1 # Standard 30s epoch math
        
        # Interval is (Next Start - Current Start). If it's the last event, it's '-'
        if i < len(events) - 1:
            interval_val = events[i+1]['start'] - ev['start']
            interval_str = f"{interval_val:.3f},"
        else:
            interval_str = "-,"
            
        # CHANGED: Use fixed-width string padding instead of messy tabs (\t)
        idx_str = f"{idx},"
        abs_str = f"{abschnitt},"
        ep_str = f"{epoch},"
        start_str = f"{ev['start']:.3f},"
        dur_str = f"{ev['duration']:.3f},"
        
        # The :< number forces the string to take up exactly that much space, aligning left.
        line_str = f"{idx_str:<4} {abs_str:<11} {ep_str:<7} {start_str:<14} {dur_str:<12} {interval_str:<14} {ev['type']}\n"
        
        formatted_event_lines.append(line_str)
    print("3. Stitching the new file together...")
    # Rebuild the file line by line
    with open(OUTPUT_TXT_PATH, 'w', encoding='latin-1') as outfile:
        i = 0
        while i < len(lines):
            # Update the Header count
            if i == table_start_idx:
                outfile.write(f"Respiratory/Apnea/Hypopnea ({len(events)} Ereignisse)\n")
                i += 1
                continue
                
            # If we are inside the old table data, skip writing it (we replace it)
            if table_start_idx < i < table_end_idx:
                # Keep the header and dashed line
                if lines[i].strip().startswith('-') or lines[i].strip().startswith('#'):
                    outfile.write(lines[i])
                
                # Once we hit the dashed line, dump all our newly formatted events!
                if lines[i].strip().startswith('-'):
                    outfile.writelines(formatted_event_lines)
            else:
                # Write everything else in the file exactly as it was
                outfile.write(lines[i])
            i += 1

    print(f"✅ SUCCESS! Updated file saved to: {OUTPUT_TXT_PATH}")

if __name__ == "__main__":
    insert_events_into_txt()