import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_opening 
from actor_critic_lstm import ActorCriticLSTM

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TYPE = 'OSA'    
NIGHT_TO_TEST = 1     
SAMPLING_RATE = 32
# ==========================================

def clean_and_stitch_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Stitched Delta-Cleaning for {TARGET_TYPE} on Night {NIGHT_TO_TEST} ---")
    
    # 1. Load the Data
    X = np.load(f'X_{NIGHT_TO_TEST}.npy')
    Y_original = np.load(f'Y_{TARGET_TYPE}_{NIGHT_TO_TEST}.npy')
    segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy')
    
    # 2. Load the Fully Trained RLHF Model
    agent = ActorCriticLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
    weights_path = f'rlhf_penta_lstm_{TARGET_TYPE}_weights.pth'
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Cannot find {weights_path}.")
        return
        
    agent.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    agent.eval() 

    ai_indices = [0, 3, 4, 5, 6, 7]
    human_indices = [0, 1, 2, 5] 

    # Prepare continuous timelines
    min_time = segment_times[0, 0]
    max_time = segment_times[-1, -1]
    total_samples = int(np.ceil((max_time - min_time) * SAMPLING_RATE)) + 1
    
    continuous_AI = np.zeros(total_samples, dtype=int)
    continuous_Doc = np.zeros(total_samples, dtype=int)
    
    filter_structure = np.ones(320) # 10-second filter
    print("\n[1/4] Running AI Inference & Stitching Timelines...")
    
    # Run Inference & Stitch
    for i in range(len(X)):
        obs = X[i][:, ai_indices].astype(np.float32)
        obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_logits, _ = agent(obs_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            greedy_action = torch.argmax(probs, dim=-1)[0].cpu().numpy()
            filtered_ai = binary_opening(greedy_action, structure=filter_structure).astype(int)
        
        target = Y_original[i].flatten()
        
        # Calculate indices for the continuous array
        s_idx = int(round((segment_times[i, 0] - min_time) * SAMPLING_RATE))
        e_idx = s_idx + len(target)
        
        # Bitwise OR perfectly handles overlaps
        continuous_AI[s_idx:e_idx] |= filtered_ai
        continuous_Doc[s_idx:e_idx] |= target

    print("[2/4] Extracting Unique Disagreements...")
    
    def get_events(arr):
        padded = np.pad(arr, (1, 1), 'constant')
        diffs = np.diff(padded)
        return list(zip(np.where(diffs == 1)[0], np.where(diffs == -1)[0]))

    ai_events = get_events(continuous_AI)
    doc_events = get_events(continuous_Doc)
    
    deltas = []
    
    # Find AI Discoveries (Doc missed it completely)
    for s, e in ai_events:
        if np.sum(continuous_Doc[s:e]) == 0:
            deltas.append({'start_idx': s, 'end_idx': e, 'type': 'AI_DISCOVERY'})
            
    # Find False Alarms (AI missed what Doc flagged)
    for s, e in doc_events:
        if np.sum(continuous_AI[s:e]) == 0:
            deltas.append({'start_idx': s, 'end_idx': e, 'type': 'DOC_ONLY'})

    # Sort sequentially by time
    deltas = sorted(deltas, key=lambda x: x['start_idx'])
    total_deltas = len(deltas)
    print(f"Found {total_deltas} UNIQUE physiological disagreements.")

    # We build the master edited array starting from the Doctor's baseline
    continuous_Y_edited = np.copy(continuous_Doc)
    seg_mid_times = segment_times[:, 480] 

    print("\n[3/4] Launching Review UI...")
    
    for idx, delta in enumerate(deltas):
        d_start = delta['start_idx']
        d_end = delta['end_idx']
        d_type = delta['type']
        
        real_start_time = min_time + (d_start / SAMPLING_RATE)
        real_end_time = min_time + (d_end / SAMPLING_RATE)
        mid_time = (real_start_time + real_end_time) / 2.0
        
        # Find single best segment
        best_seg_idx = np.argmin(np.abs(seg_mid_times - mid_time))
        
        time_axis = segment_times[best_seg_idx]
        current_x = X[best_seg_idx]
        vis_signals = current_x[:, human_indices] 
        ratio_signal = current_x[:, 6] 
        
        # Get array bounds for plotting exactly what's in this segment
        seg_s_idx = int(round((time_axis[0] - min_time) * SAMPLING_RATE))
        seg_e_idx = min(seg_s_idx + len(time_axis), total_samples)
        
        ai_pred_plot = continuous_AI[seg_s_idx:seg_e_idx]
        doc_target_plot = continuous_Doc[seg_s_idx:seg_e_idx]

        # Plotting
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        plot_color = 'red' if TARGET_TYPE == 'CA' else 'orange'
        
        axes[0].plot(time_axis, vis_signals[:, 0], color='blue', alpha=0.7)
        axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                             where=(doc_target_plot == 1), color='grey', alpha=0.3, label='Doc Baseline')
        axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                             where=(ai_pred_plot == 1), color=plot_color, alpha=0.3, label='AI Proposal')
        axes[0].set_ylabel('PFlow_Clean')
        axes[0].legend(loc='upper right')
        axes[0].set_title(f"Unique Disagreement {idx+1}/{total_deltas} | Type: {d_type}")

        axes[1].plot(time_axis, vis_signals[:, 1], color='green')
        axes[2].plot(time_axis, vis_signals[:, 2], color='purple')
        axes[3].plot(time_axis, ratio_signal, color='darkred')
        axes[4].plot(time_axis, vis_signals[:, 3], color='magenta')

        for ax in axes: ax.grid(True, alpha=0.3)
        plt.xlabel("Real Elapsed Time (Seconds)")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        print(f"\n--- Disagreement [{idx+1}/{total_deltas}] ---")
        if d_type == 'AI_DISCOVERY':
            print(f"--> AI flagged a MISSING event from {real_start_time:.1f}s to {real_end_time:.1f}s.")
        else:
            print(f"--> Doc flagged an event AI thinks is normal ({real_start_time:.1f}s to {real_end_time:.1f}s).")
            
        user_input = input("Decision: [1] AI is right, [0] Doc is right, [m] Manual bounds, [q] Quit: ").strip().lower()
        plt.close(fig) 
        
        if user_input == 'q':
            print("Quitting early. Confirmed changes will be saved.")
            break
            
        elif user_input == '1':
            if d_type == 'AI_DISCOVERY':
                continuous_Y_edited[d_start:d_end] = 1 # AI was right, add it
                print("--> Event Added.")
            else:
                continuous_Y_edited[d_start:d_end] = 0 # AI was right (no event), delete doc's label
                print("--> Event Deleted.")
                
        elif user_input == '0':
            print("--> Doc baseline kept.")
            # Do nothing, continuous_Y_edited is already a copy of the Doc baseline
            
        elif user_input == 'm':
            try:
                t_start = float(input(f"New Start (Press Enter to keep {real_start_time:.1f}): ") or real_start_time)
                t_end = float(input(f"New End (Press Enter to keep {real_end_time:.1f}): ") or real_end_time)
                
                # Clear the existing disagreement area
                continuous_Y_edited[d_start:d_end] = 0 
                
                # Set the manual boundaries
                new_s_idx = int(round((t_start - min_time) * SAMPLING_RATE))
                new_e_idx = int(round((t_end - min_time) * SAMPLING_RATE))
                continuous_Y_edited[new_s_idx:new_e_idx] = 1
                
                print(f"--> Saved manual bounds: {t_start}s - {t_end}s.")
            except ValueError:
                print("Invalid input. Doc baseline kept.")

    print("\n[4/4] Resegmenting Data into Original Format...")
    Y_final_clean = np.zeros_like(Y_original)
    
    for i in range(len(Y_original)):
        s_idx = int(round((segment_times[i, 0] - min_time) * SAMPLING_RATE))
        e_idx = s_idx + len(Y_original[i])
        
        e_idx = min(e_idx, len(continuous_Y_edited))
        segment_data = continuous_Y_edited[s_idx:e_idx]
        
        Y_final_clean[i, :len(segment_data)] = segment_data.reshape(-1, 1)

    save_path = f'Y_{TARGET_TYPE}_{NIGHT_TO_TEST}_CLEAN.npy'
    np.save(save_path, Y_final_clean)
    print(f"\n✅ [SUCCESS] Cleaned, stitched labels saved to {save_path}")

if __name__ == "__main__":
    clean_and_stitch_labels()