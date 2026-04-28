import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import sys

# ==========================================
# --- PATH RESOLUTION & IMPORTS ---
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from train_lstm import ConvLSTM
    print("✅ Successfully imported ConvLSTM from the parent directory.")
except ImportError as e:
    print(f"❌ Could not find train_lstm.py. Error: {e}")
    sys.exit()

# ==========================================
# --- CONFIGURATION ---
# ==========================================
NIGHT_ID = 12           # Pick an UNCLEANED night for the PoC
TARGET_TYPE = 'OSA'    
SAMPLING_RATE = 32
MAX_PAIRS = 150        

AI_CHANNELS = [0, 3, 4, 5, 6, 7] 
HUMAN_CHANNELS = [0, 1, 2, 5, 6] 

WEIGHTS_PATH = os.path.join(parent_dir, f'penta_lstm_{TARGET_TYPE}_weights.pth')
OUTPUT_FILE = f'dpo_preferences_n{NIGHT_ID}.npy'

# ==========================================
# --- INTERACTIVE UI HANDLER ---
# ==========================================
class ClickHandler:
    def __init__(self):
        self.choice = None
    def choose_a(self, event):
        self.choice = 'a'
        plt.close()
    def choose_b(self, event):
        self.choice = 'b'
        plt.close()
    def skip(self, event):
        self.choice = 's'
        plt.close()
    def quit(self, event):
        self.choice = 'q'
        plt.close()

# ==========================================

def extract_events(mask):
    padded = np.pad(mask, (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends))

def enforce_10s_rule(mask, sampling_rate=32):
    min_samples = 10 * sampling_rate
    cleaned_mask = np.copy(mask)
    events = extract_events(cleaned_mask)
    
    for start, end in events:
        if (end - start) < min_samples:
            cleaned_mask[start:end] = 0
    return cleaned_mask

def clinical_jitter(true_mask):
    """Generates degraded boundaries OR completely removes events (Existence Jitter)."""
    events = extract_events(true_mask)
    jittered_mask = np.copy(true_mask)
    
    for start, end in events:
        # NEW: Added 'remove_entirely' to teach the model to kill false alarms
        jitter_type = np.random.choice([
            'early_termination', 
            'late_start',       
            'merged_events',    
            'remove_entirely'      
        ], p=[0.30, 0.30, 0.15, 0.25])
        
        if jitter_type == 'remove_entirely':
            jittered_mask[start:end] = 0
            
        elif jitter_type == 'early_termination':
            early_by = np.random.randint(3*SAMPLING_RATE, 8*SAMPLING_RATE)
            if (end - early_by) > start:
                jittered_mask[end - early_by:end] = 0
                
        elif jitter_type == 'late_start':
            late_by = np.random.randint(2*SAMPLING_RATE, 5*SAMPLING_RATE)
            if (start + late_by) < end:
                jittered_mask[start:start + late_by] = 0
                
        elif jitter_type == 'merged_events':
            current_idx = events.index((start, end))
            if current_idx < len(events) - 1:
                next_start, next_end = events[current_idx + 1]
                if (next_start - end) < 15 * SAMPLING_RATE: 
                    jittered_mask[end:next_start] = 1
                    
    return jittered_mask

def run_collector():
    print(f"--- Starting Interactive DPO Data Collection for Night {NIGHT_ID} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTM().to(device)
    
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"❌ Error: Could not find weights at {WEIGHTS_PATH}")
        sys.exit()
        
    model.eval()
    
    X = np.load(os.path.join(parent_dir, 'Nights', f'X_{NIGHT_ID}.npy'))
    segment_times = np.load(os.path.join(parent_dir, 'Nights', f'segment_times_n{NIGHT_ID}.npy'))
    
    num_segments = len(X)
    preference_dataset = []
    events_graded = 0
    
    for i in range(num_segments):
        if events_graded >= MAX_PAIRS:
            break
            
        segment = X[i]
        time_axis = segment_times[i]
        
        batch_x = torch.tensor(segment[:, AI_CHANNELS], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1, :].cpu().numpy().flatten()
            
        raw_mask = (probs > 0.25).astype(int)
        pred_mask = enforce_10s_rule(raw_mask, SAMPLING_RATE)
        
        if np.sum(pred_mask) == 0:
            continue
            
        jittered_mask = clinical_jitter(pred_mask)
        if np.array_equal(pred_mask, jittered_mask):
            continue
            
        is_pred_A = np.random.choice([True, False])
        mask_A = pred_mask if is_pred_A else jittered_mask
        mask_B = jittered_mask if is_pred_A else pred_mask

        # --- UPDATED PLOT SETUP ---
        # Increased height (12) for better visibility
        fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
        plt.subplots_adjust(bottom=0.15, hspace=0.3) 
        
        vis_signals = segment[:, HUMAN_CHANNELS]
        
        titles = ['PFlow_Clean', 'Thorax', 'Abdomen', 'Effort/Flow Ratio', 'SaO2']
        colors = ['blue', 'green', 'purple', 'darkred', 'magenta']

        for idx, ax in enumerate(axes):
            ax.plot(time_axis, vis_signals[:, idx], color=colors[idx], alpha=0.8)
            ax.set_ylabel(titles[idx])
            ax.grid(True, alpha=0.3)
            
            # --- THE FIX: ENFORCE PERSPECTIVE ---
            # Using -2.2 to 2.2 to ensure -1 and 1 are clearly visible with some context
            if titles[idx] != 'SaO2': # SaO2 is usually 0 to -10, keep it auto or set specifically
                ax.set_ylim(-2.2, 2.2)
                ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)
                ax.axhline(-1.0, color='black', linestyle='--', alpha=0.3)
            else:
                # SaO2 specifically often needs a different range
                ax.set_ylim(-10, 2)

            ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1], 
                            where=(mask_A == 1), color='cyan', alpha=0.3, label='Box A' if idx == 0 else "")
            ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1], 
                            where=(mask_B == 1), color='orange', alpha=0.3, label='Box B' if idx == 0 else "")

        axes[0].set_title(f"Event {events_graded+1}/{MAX_PAIRS} | Is it an Apnea? Where are the bounds?")
        axes[0].legend(loc='upper right')
        
        # --- BUTTON SETUP ---
        handler = ClickHandler()
        ax_a = plt.axes([0.25, 0.03, 0.1, 0.05])
        ax_b = plt.axes([0.40, 0.03, 0.1, 0.05])
        ax_skip = plt.axes([0.55, 0.03, 0.1, 0.05])
        ax_quit = plt.axes([0.70, 0.03, 0.1, 0.05])
        
        btn_a = Button(ax_a, 'Box A (Cyan)', color='cyan')
        btn_b = Button(ax_b, 'Box B (Orange)', color='orange')
        btn_skip = Button(ax_skip, 'Skip')
        btn_quit = Button(ax_quit, 'Quit', color='lightcoral')
        
        btn_a.on_clicked(handler.choose_a)
        btn_b.on_clicked(handler.choose_b)
        btn_skip.on_clicked(handler.skip)
        btn_quit.on_clicked(handler.quit)

        plt.show() 
        
        if handler.choice == 'q':
            break
        elif handler.choice == 's':
            continue
        elif handler.choice in ['a', 'b']:
            chosen_mask = mask_A if handler.choice == 'a' else mask_B
            rejected_mask = mask_B if handler.choice == 'a' else mask_A
            
            preference_dataset.append({
                'context_signal': segment[:, AI_CHANNELS], 
                'chosen_mask': chosen_mask,
                'rejected_mask': rejected_mask
            })
            events_graded += 1
            print(f"--> Saved preference {events_graded}/{MAX_PAIRS}.")
            
    if len(preference_dataset) > 0:
        np.save(OUTPUT_FILE, preference_dataset)
        print(f"\n✅ SUCCESS! Saved {len(preference_dataset)} preference pairs to {OUTPUT_FILE}.")
    else:
        print("\nNo preferences saved.")

if __name__ == "__main__":
    run_collector()