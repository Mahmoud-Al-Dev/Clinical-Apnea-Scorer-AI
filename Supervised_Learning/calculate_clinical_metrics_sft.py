import torch
import numpy as np
import os
from scipy.ndimage import label
from train_lstm import ConvLSTM 

def apply_cleanup_filter(predictions, min_length_frames=320):
    cleaned = np.copy(predictions)
    labeled_array, num_features = label(cleaned == 1)
    for i in range(1, num_features + 1):
        if np.sum(labeled_array == i) < min_length_frames:
            cleaned[labeled_array == i] = 0
    return cleaned

def evaluate_clinical_events(predictions, ground_truth, min_length=320, overlap_threshold=0.10):
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
            
    # Standard (Misleading) Metrics - Penalizes AI for finding unlabelled events
    standard_precision = confirmed_ai_events / num_pred if num_pred > 0 else 0.0
    standard_f1 = 2 * (standard_precision * recall) / (standard_precision + recall) if (standard_precision + recall) > 0 else 0.0
        
    return {
        "doctor_events_total": num_true,
        "ai_caught_events": matched_doctor_events,
        "ai_total_predictions": num_pred,
        "ai_confirmed_predictions": confirmed_ai_events,
        "ai_unlabeled_discoveries": unlabeled_ai_discoveries,
        "recall_sensitivity": float(recall),
        "standard_precision": float(standard_precision),
        "standard_f1_score": float(standard_f1)
    }

def evaluate_full_night(model, night_num, target_type, device):
    model.eval()
    
    # 1. Load Data
    X = np.load(f'Nights/X_{night_num}.npy')
    
    # SMART LOAD: Use Silver Standard if available to get mathematically accurate metrics
    silver_path = f'Nights/Y_{target_type}_{night_num}_SILVER.npy'
    if os.path.exists(silver_path):
        Y_true = np.load(silver_path)
    else:
        Y_true = np.load(f'Nights/Y_{target_type}_{night_num}.npy')
    
    ai_indices = [0, 3, 4, 5, 6, 7]
    num_segments = len(X)
    batch_size = 64
    probs = np.zeros((num_segments, 960))
    
    # 2. Batch Inference
    with torch.no_grad():
        for i in range(0, num_segments, batch_size):
            end_idx = min(i + batch_size, num_segments)
            batch_x = torch.tensor(X[i:end_idx, :, ai_indices], dtype=torch.float32).to(device)
            logits = model(batch_x)
            probs[i:end_idx] = torch.softmax(logits, dim=1)[:, 1, :].cpu().numpy()

    # 3. Stitching overlaps
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

    # 4. Calculate metrics
    return evaluate_clinical_events(full_classes, full_y)

def run_multi_night_evaluation(model, test_nights, target_type, device):
    print(f"\n==================================================")
    print(f"🚀 MULTI-NIGHT EVALUATION REPORT (TARGET: {target_type}) 🚀")
    print(f"==================================================")
    
    # Aggregate Counters
    total_doc_events = 0
    total_ai_caught = 0
    total_ai_preds = 0
    total_ai_confirmed = 0
    total_ai_discoveries = 0
    
    print("\n--- PER-NIGHT BREAKDOWN ---")
    
    for night in test_nights:
        res = evaluate_full_night(model, night, target_type, device)
        
        # Add to global aggregates
        total_doc_events += res["doctor_events_total"]
        total_ai_caught += res["ai_caught_events"]
        total_ai_preds += res["ai_total_predictions"]
        total_ai_confirmed += res["ai_confirmed_predictions"]
        total_ai_discoveries += res["ai_unlabeled_discoveries"]
        
        # Print Single Night Snapshot
        recall_pct = res['recall_sensitivity'] * 100
        f1_pct = res['standard_f1_score'] * 100
        prec_pct = res['standard_precision'] * 100
        print(f"Night {night:02d} | F1: {f1_pct:05.2f}% | Recall: {recall_pct:05.2f}% | Prec: {prec_pct:05.2f}% | "
              f"Truth: {res['doctor_events_total']:<4} | AI Caught: {res['ai_caught_events']:<4} | "
              f"New AI Discoveries: {res['ai_unlabeled_discoveries']}")

    # Calculate Global Metrics
    global_recall = total_ai_caught / total_doc_events if total_doc_events > 0 else 0.0
    global_precision = total_ai_confirmed / total_ai_preds if total_ai_preds > 0 else 0.0
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
    
    print("\n==================================================")
    print(f"--- GLOBAL AGGREGATE STATISTICS (Nights: {test_nights}) ---")
    print(f"Total True Events (Doctor):      {total_doc_events}")
    print(f"Total AI Caught:                 {total_ai_caught}")
    print(f"Total AI Predictions Drawn:      {total_ai_preds}")
    print(f"Confirmed AI Predictions:        {total_ai_confirmed}")
    print(f"Unlabeled AI Discoveries:        {total_ai_discoveries} (Pending Review)")
    print("--------------------------------------------------")
    print(f"🏆 GLOBAL RECALL (Sensitivity):  {global_recall:.4f}")
    print(f"📊 GLOBAL PRECISION (Standard):  {global_precision:.4f}")
    print(f"🎯 GLOBAL F1 SCORE:              {global_f1:.4f}")
    print("==================================================\n")

# ==========================================
# --- STANDALONE TESTER ---
# ==========================================
if __name__ == "__main__":
    TEST_NIGHTS = [11,12,13]  
    TEST_TARGET = 'OSA'  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
    
    weights_path = f'penta_lstm_{TEST_TARGET}_weights.pth'
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"✅ Successfully loaded SFT weights from {weights_path}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {weights_path}.")
        exit()
    
    run_multi_night_evaluation(model, TEST_NIGHTS, TEST_TARGET, device)