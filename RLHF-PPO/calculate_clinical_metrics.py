import torch
import numpy as np
from scipy.ndimage import label
from actor_critic_lstm import ActorCriticLSTM

def apply_cleanup_filter(predictions, min_length_frames=320):
    cleaned = np.copy(predictions)
    labeled_array, num_features = label(cleaned == 1)
    for i in range(1, num_features + 1):
        if np.sum(labeled_array == i) < min_length_frames:
            cleaned[labeled_array == i] = 0
    return cleaned

def evaluate_clinical_events(predictions, ground_truth, min_length=320, overlap_threshold=0.30):
    preds_clean = apply_cleanup_filter(predictions, min_length)
    
    true_events, num_true = label(ground_truth == 1)
    pred_events, num_pred = label(preds_clean == 1)
    
    matched_doctor_events = 0
    for i in range(1, num_true + 1):
        doc_mask = (true_events == i)
        overlap = np.sum(doc_mask & (preds_clean == 1))
        if overlap > (np.sum(doc_mask) * overlap_threshold):
            matched_doctor_events += 1
            
    recall = matched_doctor_events / num_true if num_true > 0 else 0.0
    
    confirmed_ai_events = 0
    unlabeled_ai_discoveries = 0
    
    for i in range(1, num_pred + 1):
        ai_mask = (pred_events == i)
        overlap = np.sum(ai_mask & (ground_truth == 1))
        if overlap > 0:
            confirmed_ai_events += 1
        else:
            unlabeled_ai_discoveries += 1
            
    precision = confirmed_ai_events / num_pred if num_pred > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return {
        "doctor_events_total": num_true,
        "ai_caught_events": matched_doctor_events,
        "recall_sensitivity": float(recall),
        "ai_total_predictions": num_pred,
        "ai_confirmed_predictions": confirmed_ai_events,
        "ai_unlabeled_discoveries": unlabeled_ai_discoveries,
        "precision": float(precision),
        "f1_score": float(f1)
    }

def evaluate_full_night(model, night_num, target_type, device):
    """
    Runs inference on the full night, stitches the overlapping arrays, 
    and calculates clinical metrics. Automatically detects SFT vs RLHF models!
    """
    model.eval()
    
    # 1. Load Data
    X = np.load(f'X_{night_num}.npy')
    Y_true = np.load(f'Y_{target_type}_{night_num}.npy')
    
    # Define the 6 AI indices to slice the array
    ai_indices = [0, 3, 4, 5, 6, 7]
    
    num_segments = len(X)
    batch_size = 64
    probs = np.zeros((num_segments, 960))
    
    # 2. Batch Inference
    with torch.no_grad():
        for i in range(0, num_segments, batch_size):
            end_idx = min(i + batch_size, num_segments)
            
            # Slice the batch to only include the AI channels
            batch_x = torch.tensor(X[i:end_idx, :, ai_indices], dtype=torch.float32).to(device)
            
            outputs = model(batch_x)
            
            # --- THE MAGIC FIX: Detect Model Type ---
            if isinstance(outputs, tuple):
                # It's the RLHF ActorCriticLSTM (returns action_logits, state_value)
                logits = outputs[0]
                # Shape is (Batch, 960, 2), so we softmax on dim=-1
                probs[i:end_idx] = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            else:
                # It's the SFT ConvLSTM (returns only logits)
                logits = outputs
                # Shape is (Batch, 2, 960), so we softmax on dim=1
                probs[i:end_idx] = torch.softmax(logits, dim=1)[:, 1, :].cpu().numpy()

    # 3. Stitching overlaps (640 step, 960 window)
    win_samples = 960
    step_samples = 640
    total_samples = step_samples * (num_segments - 1) + win_samples

    full_y = np.zeros(total_samples)
    full_probs = np.zeros(total_samples)
    overlap_counts = np.zeros(total_samples)

    for i in range(num_segments):
        start_idx = i * step_samples
        end_idx = start_idx + win_samples
        
        full_y[start_idx:end_idx] = np.maximum(full_y[start_idx:end_idx], Y_true[i].flatten())
        full_probs[start_idx:end_idx] += probs[i]
        overlap_counts[start_idx:end_idx] += 1

    full_probs /= overlap_counts
    full_classes = (full_probs > 0.5).astype(int)

    # 4. Run the overlap math
    return evaluate_clinical_events(full_classes, full_y)


# ==========================================
# --- STANDALONE TESTER ---
# ==========================================
if __name__ == "__main__":
    TEST_NIGHT = 1
    TEST_TARGET = 'OSA'
    
    # CHANGE THIS to 'SFT' if you want to test your raw ConvLSTM weights!
    EVAL_MODE = 'RLHF' 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if EVAL_MODE == 'RLHF':
        model = ActorCriticLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
        weights_path = f'rlhf_penta_lstm_{TEST_TARGET}_weights.pth'
    else:
        # Import whatever you named your SFT class here (e.g., ConvLSTM)
        from train_lstm import ConvLSTM 
        model = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
        weights_path = f'penta_lstm_{TEST_TARGET}_weights.pth'

    print(f"Loading {EVAL_MODE} weights from {weights_path}...")
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except RuntimeError as e:
        print("\n❌ ARCHITECTURE MISMATCH ERROR!")
        print("Did you try to load old RLHF weights into the new CNN-enabled ActorCriticLSTM?")
        print("You need to run train_rlhf_ppo.py first to generate the new weights!\n")
        raise e
    
    print(f"Running Full Night Evaluation for {TEST_TARGET} on Night {TEST_NIGHT}...")
    results = evaluate_full_night(model, TEST_NIGHT, TEST_TARGET, device)
    
    print("\n--- RLHF Event-Based Validation Results ---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")