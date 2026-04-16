import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'CA'    
NIGHT_ID = 6
SAMPLING_RATE = 32 # Assuming 32Hz based on standard 960-sample 30s windows

# The file you want to load and fix:
INPUT_LABELS_PATH = f'Nights\Y_{TARGET_TYPE}_{NIGHT_ID}.npy'
# We save to a new file so we don't accidentally destroy your previous work
OUTPUT_LABELS_PATH = f'Nights\Y_{TARGET_TYPE}_{NIGHT_ID}_ADJUSTED.npy'
# ==========================================

def adjust_stitched_boundaries():
    print(f"--- Starting Stitched Boundary Adjustment for {TARGET_TYPE} on Night {NIGHT_ID} ---")
    
    if not os.path.exists(INPUT_LABELS_PATH):
        print(f"ERROR: Cannot find {INPUT_LABELS_PATH}.")
        return

    # 1. Load the Data
    X = np.load(f'Nights\X_{NIGHT_ID}.npy')
    Y_current = np.load(INPUT_LABELS_PATH)
    segment_times = np.load(f'Nights\segment_times_n{NIGHT_ID}.npy')
    
    human_indices = [0, 1, 2, 5] # PFlow, Thorax, Abdomen, SaO2
    
    # 2. Stitch Y into a continuous timeline to eliminate overlaps
    min_time = segment_times[0, 0]
    max_time = segment_times[-1, -1]
    total_samples = int(np.ceil((max_time - min_time) * SAMPLING_RATE)) + 1
    
    continuous_Y = np.zeros(total_samples, dtype=int)
    
    for i in range(len(Y_current)):
        start_idx = int(round((segment_times[i, 0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(Y_current[i])
        # Logical OR merges the overlaps perfectly
        continuous_Y[start_idx:end_idx] |= Y_current[i].flatten()

    # 3. Extract unique events from the continuous timeline
    padded = np.pad(continuous_Y, (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    unique_events = []
    for s, e in zip(starts, ends):
        unique_events.append({
            'start_time': min_time + (s / SAMPLING_RATE),
            'end_time': min_time + (e / SAMPLING_RATE)
        })

    total_events = len(unique_events)
    print(f"Found {total_events} UNIQUE {TARGET_TYPE} events after merging overlaps. Starting review...")

    # We will build a brand new continuous array based ONLY on your manual confirmations
    continuous_Y_edited = np.zeros_like(continuous_Y)

    # 4. UI Loop - Iterating over EVENTS, not segments
    for idx, event in enumerate(unique_events):
        current_start = event['start_time']
        current_end = event['end_time']
        mid_time = (current_start + current_end) / 2.0
        
        # Find the single best segment to display this event (the one whose center is closest to the event)
        seg_mid_times = segment_times[:, 480] # Index 480 is the middle of the 960-sample window
        best_seg_idx = np.argmin(np.abs(seg_mid_times - mid_time))
        
        time_axis = segment_times[best_seg_idx]
        current_x = X[best_seg_idx]
        vis_signals = current_x[:, human_indices] 
        ratio_signal = current_x[:, 6] 

        # Plotting
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        plot_color = 'red' if TARGET_TYPE == 'CA' else 'orange'
        
        # We create a local mask just for plotting the current bounds
        plot_mask = (time_axis >= current_start) & (time_axis <= current_end)
        
        axes[0].plot(time_axis, vis_signals[:, 0], color='blue', alpha=0.7)
        axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                             where=plot_mask, color=plot_color, alpha=0.4, label=f'Current {TARGET_TYPE}')
        axes[0].set_ylabel('PFlow_Clean')
        axes[0].legend(loc='upper right')
        axes[0].set_title(f"Unique Event {idx+1}/{total_events} | Displaying in Best Segment: {best_seg_idx}")

        axes[1].plot(time_axis, vis_signals[:, 1], color='green')
        axes[1].fill_between(time_axis, np.min(vis_signals[:, 1]), np.max(vis_signals[:, 1]), where=plot_mask, color=plot_color, alpha=0.2)
        
        axes[2].plot(time_axis, vis_signals[:, 2], color='purple')
        axes[2].fill_between(time_axis, np.min(vis_signals[:, 2]), np.max(vis_signals[:, 2]), where=plot_mask, color=plot_color, alpha=0.2)
        
        axes[3].plot(time_axis, ratio_signal, color='darkred')
        axes[3].fill_between(time_axis, np.min(ratio_signal), np.max(ratio_signal), where=plot_mask, color=plot_color, alpha=0.2)
        
        axes[4].plot(time_axis, vis_signals[:, 3], color='magenta')
        axes[4].fill_between(time_axis, np.min(vis_signals[:, 3]), np.max(vis_signals[:, 3]), where=plot_mask, color=plot_color, alpha=0.2)

        for ax in axes: ax.grid(True, alpha=0.3)
        plt.xlabel("Real Elapsed Time (Seconds)")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        print(f"\n--- Event [{idx+1}/{total_events}] ---")
        print(f"Current bounds: Start={current_start:.1f}s, End={current_end:.1f}s")
        
        user_input = input("Action: [Enter] Keep as is, [m] Manual adjust, [d] Delete event, [q] Quit: ").strip().lower()
        plt.close(fig) 
        
        if user_input == 'q':
            print("Quitting early. Any confirmed changes up to this point will be saved.")
            break
            
        elif user_input == 'd':
            print("--> Event deleted.")
            # Do nothing to continuous_Y_edited, leaving it as 0s
            
        elif user_input == 'm':
            try:
                t_start_input = input(f"New Start (Press Enter to keep {current_start:.1f}): ").strip()
                t_end_input = input(f"New End (Press Enter to keep {current_end:.1f}): ").strip()
                
                final_start = float(t_start_input) if t_start_input else current_start
                final_end = float(t_end_input) if t_end_input else current_end
                
                # Apply new boundaries to the master continuous array
                s_idx = int(round((final_start - min_time) * SAMPLING_RATE))
                e_idx = int(round((final_end - min_time) * SAMPLING_RATE))
                continuous_Y_edited[s_idx:e_idx] = 1
                
                print(f"--> Saved adjusted bounds: {final_start}s - {final_end}s.")
            except ValueError:
                print("Invalid input. Reverting to original boundaries.")
                s_idx = int(round((current_start - min_time) * SAMPLING_RATE))
                e_idx = int(round((current_end - min_time) * SAMPLING_RATE))
                continuous_Y_edited[s_idx:e_idx] = 1
                
        else:
            print("--> Event kept as is.")
            s_idx = int(round((current_start - min_time) * SAMPLING_RATE))
            e_idx = int(round((current_end - min_time) * SAMPLING_RATE))
            continuous_Y_edited[s_idx:e_idx] = 1

    # 5. Slice the continuous array back into overlapping segments
    print("\nRe-segmenting data into original overlapping format...")
    Y_adjusted = np.zeros_like(Y_current)
    
    for i in range(len(Y_current)):
        start_idx = int(round((segment_times[i, 0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(Y_current[i])
        
        # Ensure we don't go out of bounds
        end_idx = min(end_idx, len(continuous_Y_edited))
        segment_data = continuous_Y_edited[start_idx:end_idx]
        
        Y_adjusted[i, :len(segment_data)] = segment_data.reshape(-1, 1)

    # 6. Save the final file
    np.save(OUTPUT_LABELS_PATH, Y_adjusted)
    print(f"[SUCCESS] Adjusted overlapping labels saved to {OUTPUT_LABELS_PATH}!")

if __name__ == "__main__":
    adjust_stitched_boundaries()