import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'OSA'    # Change to 'CA' when needed
NIGHT_ID = 16
SAMPLING_RATE = 32 

# 1. SMART LOADING: Try Silver first, fallback to Raw
silver_path = f'Nights\Y_{TARGET_TYPE}_{NIGHT_ID}_SILVER.npy'
raw_path = f'Nights\Y_{TARGET_TYPE}_{NIGHT_ID}.npy'

if os.path.exists(silver_path):
    print(f"📂 Found SILVER standard. Loading all AI discoveries...")
    INPUT_LABELS_PATH = silver_path
else:
    print(f"⚠️ No SILVER standard found. Falling back to original labels...")
    INPUT_LABELS_PATH = raw_path
    
OUTPUT_LABELS_PATH = f'Nights\Y_{TARGET_TYPE}_{NIGHT_ID}_ADJUSTED.npy'
# ==========================================

def adjust_stitched_boundaries():
    print(f"--- Starting Stitched Boundary Adjustment for {TARGET_TYPE} on Night {NIGHT_ID} ---")
    
    if not os.path.exists(INPUT_LABELS_PATH):
        print(f"ERROR: Cannot find {INPUT_LABELS_PATH}.")
        return

    X = np.load(f'Nights\X_{NIGHT_ID}.npy')
    Y_current = np.load(INPUT_LABELS_PATH)
    segment_times = np.load(f'Nights\segment_times_n{NIGHT_ID}.npy')
    
    human_indices = [0, 1, 2, 5] 
    
    min_time = segment_times[0, 0]
    max_time = segment_times[-1, -1]
    total_samples = int(np.ceil((max_time - min_time) * SAMPLING_RATE)) + 1
    
    continuous_Y = np.zeros(total_samples, dtype=int)
    
    for i in range(len(Y_current)):
        start_idx = int(round((segment_times[i, 0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(Y_current[i])
        continuous_Y[start_idx:end_idx] |= Y_current[i].flatten()

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
    print(f"Found {total_events} UNIQUE {TARGET_TYPE} events. Starting review...")

    # --- BUG FIX: Start with a perfect clone so 'q' doesn't delete data ---
    continuous_Y_edited = np.copy(continuous_Y)

    for idx, event in enumerate(unique_events):
        current_start = event['start_time']
        current_end = event['end_time']
        mid_time = (current_start + current_end) / 2.0
        
        seg_mid_times = segment_times[:, 480] 
        best_seg_idx = np.argmin(np.abs(seg_mid_times - mid_time))
        
        time_axis = segment_times[best_seg_idx]
        current_x = X[best_seg_idx]
        vis_signals = current_x[:, human_indices] 
        ratio_signal = current_x[:, 6] 

        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        plot_color = 'red' if TARGET_TYPE == 'CA' else 'orange'
        
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
            print("Quitting early. All unreviewed events will be kept exactly as they are.")
            break
            
        elif user_input == 'd':
            print("--> Event deleted.")
            s_idx = int(round((current_start - min_time) * SAMPLING_RATE))
            e_idx = int(round((current_end - min_time) * SAMPLING_RATE))
            continuous_Y_edited[s_idx:e_idx] = 0 # Actively erase it
            
        elif user_input == 'm':
            try:
                t_start_input = input(f"New Start (Press Enter to keep {current_start:.1f}): ").strip()
                t_end_input = input(f"New End (Press Enter to keep {current_end:.1f}): ").strip()
                
                final_start = float(t_start_input) if t_start_input else current_start
                final_end = float(t_end_input) if t_end_input else current_end
                
                # Erase old boundaries first
                s_idx_old = int(round((current_start - min_time) * SAMPLING_RATE))
                e_idx_old = int(round((current_end - min_time) * SAMPLING_RATE))
                continuous_Y_edited[s_idx_old:e_idx_old] = 0
                
                # Apply new boundaries
                s_idx_new = int(round((final_start - min_time) * SAMPLING_RATE))
                e_idx_new = int(round((final_end - min_time) * SAMPLING_RATE))
                continuous_Y_edited[s_idx_new:e_idx_new] = 1
                
                print(f"--> Saved adjusted bounds: {final_start}s - {final_end}s.")
            except ValueError:
                print("Invalid input. Event kept as is.")
        else:
            print("--> Event kept as is.")

    print("\nRe-segmenting data into original overlapping format...")
    Y_adjusted = np.zeros_like(Y_current)
    
    for i in range(len(Y_current)):
        start_idx = int(round((segment_times[i, 0] - min_time) * SAMPLING_RATE))
        end_idx = start_idx + len(Y_current[i])
        
        end_idx = min(end_idx, len(continuous_Y_edited))
        segment_data = continuous_Y_edited[start_idx:end_idx]
        
        Y_adjusted[i, :len(segment_data)] = segment_data.reshape(-1, 1)

    np.save(OUTPUT_LABELS_PATH, Y_adjusted)
    print(f"[SUCCESS] Adjusted overlapping labels saved to {OUTPUT_LABELS_PATH}!")

if __name__ == "__main__":
    adjust_stitched_boundaries()