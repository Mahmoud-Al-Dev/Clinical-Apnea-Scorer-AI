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

def evaluate_clinical_events(predictions, ground_truth, min_length=250, overlap_threshold=0.30):
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

def evaluate_full_night(model, night_num, target_type, device, input_channels=6):
    """
    Runs inference on the full night, stitches the overlapping arrays, 
    and calculates clinical metrics.
    """
    model.eval()
    
    # 1. Load Data
    X = np.load(f'X_{night_num}.npy')
    Y_true = np.load(f'Y_{target_type}_{night_num}.npy')
    
    num_segments = len(X)
    batch_size = 64
    probs = np.zeros((num_segments, 960))
    
    # 2. Batch Inference
    with torch.no_grad():
        for i in range(0, num_segments, batch_size):
            end_idx = min(i + batch_size, num_segments)
            batch_x = torch.tensor(X[i:end_idx], dtype=torch.float32).to(device)
            logits, _ = model(batch_x)
            probs[i:end_idx] = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()

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
    # Test your actual files!
    TEST_NIGHT = 2
    TEST_TARGET = 'CA'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your actual RLHF weights
    model = ActorCriticLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
    weights_path = f'rlhf_penta_lstm_{TEST_TARGET}_weights.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    print(f"Running Full Night Evaluation for {TEST_TARGET} on Night {TEST_NIGHT}...")
    results = evaluate_full_night(model, TEST_NIGHT, TEST_TARGET, device)
    
    print("\n--- Event-Based Validation Results ---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")