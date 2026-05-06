import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'CA'    
NIGHT_ID = 25
SAMPLING_RATE = 32 

silver_path = f'Nights\\Y_{TARGET_TYPE}_{NIGHT_ID}_SILVER.npy'
raw_path = f'Nights\\Y_{TARGET_TYPE}_{NIGHT_ID}.npy'

if os.path.exists(silver_path):
    print(f"📂 Found SILVER standard. Loading all AI discoveries...")
    INPUT_LABELS_PATH = silver_path
else:
    print(f"⚠️ No SILVER standard found. Falling back to original labels...")
    INPUT_LABELS_PATH = raw_path
    
OUTPUT_LABELS_PATH = f'Nights\\Y_{TARGET_TYPE}_{NIGHT_ID}_ADJUSTED.npy'
# ==========================================

def adjust_stitched_boundaries():
    print(f"--- Starting Interactive Boundary Adjustment for {TARGET_TYPE} ---")
    
    if not os.path.exists(INPUT_LABELS_PATH):
        print(f"ERROR: Cannot find {INPUT_LABELS_PATH}.")
        return

    X = np.load(f'Nights\\X_{NIGHT_ID}.npy')
    Y_current = np.load(INPUT_LABELS_PATH)
    segment_times = np.load(f'Nights\\segment_times_n{NIGHT_ID}.npy')
    
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
    
    unique_events = [{'start_time': min_time + (s / SAMPLING_RATE), 'end_time': min_time + (e / SAMPLING_RATE)} for s, e in zip(starts, ends)]
    total_events = len(unique_events)
    continuous_Y_edited = np.copy(continuous_Y)

    for idx, event in enumerate(unique_events):
        current_start, current_end = event['start_time'], event['end_time']
        mid_time = (current_start + current_end) / 2.0
        best_seg_idx = np.argmin(np.abs(segment_times[:, 480] - mid_time))
        
        time_axis = segment_times[best_seg_idx]
        vis_signals = X[best_seg_idx][:, human_indices] 
        ratio_signal = X[best_seg_idx][:, 6] 

        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        plot_color = 'red' if TARGET_TYPE == 'CA' else 'orange'
        plot_mask = (time_axis >= current_start) & (time_axis <= current_end)
        
        axes[0].plot(time_axis, vis_signals[:, 0], color='blue', alpha=0.7)
        axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), where=plot_mask, color=plot_color, alpha=0.4)
        axes[0].set_title(f"Event {idx+1}/{total_events} | Action: [Enter] Keep, [m] Manual Click, [d] Delete, [q] Quit")
        axes[1].plot(time_axis, vis_signals[:, 1], color='green')
        axes[2].plot(time_axis, vis_signals[:, 2], color='purple')
        axes[3].plot(time_axis, ratio_signal, color='darkred')
        axes[4].plot(time_axis, vis_signals[:, 3], color='magenta')

        for ax in axes: ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        
        user_input = input(f"Action ({current_start:.1f}s-{current_end:.1f}s): ").strip().lower()
        
        if user_input == 'q':
            plt.close(fig)
            break
        elif user_input == 'd':
            s_idx, e_idx = int(round((current_start - min_time) * SAMPLING_RATE)), int(round((current_end - min_time) * SAMPLING_RATE))
            continuous_Y_edited[s_idx:e_idx] = 0
            print("--> Deleted.")
            plt.close(fig)
        elif user_input == 'm':
            print("🖱️  CLICK START and END points on the graph...")
            # ginput waits for 2 clicks on the plot
            clicks = plt.ginput(2, timeout=-1) 
            if len(clicks) == 2:
                t1, t2 = clicks[0][0], clicks[1][0]
                final_start, final_end = min(t1, t2), max(t1, t2)
                
                # Erase old, apply new
                s_old, e_old = int(round((current_start - min_time) * SAMPLING_RATE)), int(round((current_end - min_time) * SAMPLING_RATE))
                continuous_Y_edited[s_old:e_old] = 0
                s_new, e_new = int(round((final_start - min_time) * SAMPLING_RATE)), int(round((final_end - min_time) * SAMPLING_RATE))
                continuous_Y_edited[s_new:e_new] = 1
                print(f"--> Adjusted: {final_start:.1f}s to {final_end:.1f}s")
            plt.close(fig)
        else:
            print("--> Kept.")
            plt.close(fig)

    # Re-segmenting back to overlapping format
    Y_adjusted = np.zeros_like(Y_current)
    for i in range(len(Y_current)):
        start_idx = int(round((segment_times[i, 0] - min_time) * SAMPLING_RATE))
        end_idx = min(start_idx + len(Y_current[i]), len(continuous_Y_edited))
        Y_adjusted[i, :end_idx-start_idx] = continuous_Y_edited[start_idx:end_idx].reshape(-1, 1)

    np.save(OUTPUT_LABELS_PATH, Y_adjusted)
    print(f"DONE. Saved to {OUTPUT_LABELS_PATH}")

if __name__ == "__main__":
    adjust_stitched_boundaries()