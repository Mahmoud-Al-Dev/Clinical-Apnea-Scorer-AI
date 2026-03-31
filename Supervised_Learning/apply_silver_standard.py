import torch
import numpy as np
from scipy.ndimage import label 
from train_lstm import ConvLSTM 

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TYPE = 'CA'  
NIGHT_ID = 3
AUTO_CORRECT_LIMIT = 80 
# ==========================================

print(f"Creating Silver Standard for {TARGET_TYPE} Night {NIGHT_ID}...")

X = np.load(f'X_{NIGHT_ID}.npy')
Y_original = np.load(f'Y_{TARGET_TYPE}_{NIGHT_ID}.npy')
flags = np.load(f'cleanlab_flags_{TARGET_TYPE}_n{NIGHT_ID}.npy')

Y_silver = np.copy(Y_original)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
model.load_state_dict(torch.load(f'penta_lstm_{TARGET_TYPE}_weights.pth', map_location=device, weights_only=True))
model.eval()

ai_indices = [0, 3, 4, 5, 6, 7] 
corrections_made = 0

for i in range(AUTO_CORRECT_LIMIT):
    seg_idx = int(flags[i])
    
    raw_segment = X[seg_idx]
    ai_obs = torch.tensor(raw_segment[:, ai_indices], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(ai_obs)
        probs = torch.softmax(logits, dim=1)[0, 1, :].cpu().numpy()
        
        # 1. Convert to binary
        ai_preds = (probs > 0.5).astype(int) 
        
        # 2. APPLY 10-SECOND CLEANUP FILTER
        labeled_array, num_features = label(ai_preds)
        for j in range(1, num_features + 1):
            if np.sum(labeled_array == j) < 320: # 320 frames = 10 seconds at 32Hz
                ai_preds[labeled_array == j] = 0
                
        # 3. Reshape safely to match Y_original
        ai_preds = ai_preds.reshape(960, 1)
        
    Y_silver[seg_idx] = ai_preds
    corrections_made += 1

print(f"\n✅ Successfully auto-corrected {corrections_made} obvious clinical errors (with 10s filter!).")

output_filename = f'Y_{TARGET_TYPE}_{NIGHT_ID}_SILVER.npy'
np.save(output_filename, Y_silver)
print(f"💾 Saved as '{output_filename}'")