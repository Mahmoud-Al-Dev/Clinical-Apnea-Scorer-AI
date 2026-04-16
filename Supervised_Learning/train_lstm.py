import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy 
import os
import mlflow

# =================================================================
# --- USER CONTROLS ---
# =================================================================
TARGET_TYPE = 'OSA' 

# 1. Dataset Splits
TRAIN_NIGHTS = [1, 2,3, 4, 5, 6, 7] 
VAL_NIGHT = 3          # Only used if USE_VALIDATION is True
TEST_NIGHTS = [2,3]   # Nights to evaluate automatically after training

# 2. Training Settings
USE_VALIDATION = False 
MAX_EPOCHS = 30
PATIENCE = 20          
LEARNING_RATE = 0.001

# 3. PU Learning & Loss Weights
CLASS_WEIGHT_NORMAL = 1.0
CLASS_WEIGHT_APNEA = 4.0
PU_DISCOUNT = 0.40
# =================================================================

class MultiNightApneaDataset(Dataset):
    def __init__(self, nights_list, target_type, folder="Nights"): 
        x_list = []
        y_list = []
        
        for night in nights_list:
            print(f"Loading Night {night} from '{folder}' folder...")
            if night == 3:
                y_file = os.path.join(folder, f'Y_{target_type}_{night}_SILVER.npy')
                try:
                    y_data = np.load(y_file)
                except FileNotFoundError:
                    print(f"  [!] Silver Standard not found. Using original labels.")
                    y_file = os.path.join(folder, f'Y_{target_type}_{night}.npy')
                    y_data = np.load(y_file)
            else:
                y_file = os.path.join(folder, f'Y_{target_type}_{night}.npy')
                y_data = np.load(y_file)
                
            x_file = os.path.join(folder, f'X_{night}.npy')
            x_data = np.load(x_file)
            
            x_list.append(x_data)
            y_list.append(y_data)
            
        self.x = torch.tensor(np.concatenate(x_list, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.long).squeeze(-1)
        self.ai_indices = [0, 3, 4, 5, 6, 7]
        
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx, :, self.ai_indices], self.y[idx]

class ConvLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(ConvLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=32, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=True 
        )
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_features)
        predictions = self.fc(lstm_out) 
        predictions = predictions.permute(0, 2, 1)
        return predictions

class SimulatedPULoss(nn.Module):
    def __init__(self, class_weights, pu_discount=0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.pu_discount = pu_discount

    def forward(self, logits, targets):
        base_loss = self.ce(logits, targets) 
        probs = torch.softmax(logits, dim=1)[:, 1, :]
        discount_mask = torch.where((targets == 0) & (probs > 0.5), self.pu_discount, 1.0)
        final_loss = base_loss * discount_mask
        return final_loss.mean()
    
def train_model():
    # Set up MLflow Experiment
    mlflow.set_experiment(f"Apnea_SFT_{TARGET_TYPE}")
    
    with mlflow.start_run():
        # --- 1. LOG HYPERPARAMETERS TO MLFLOW ---
        mlflow.log_params({
            "target_type": TARGET_TYPE,
            "train_nights": str(TRAIN_NIGHTS),
            "test_nights": str(TEST_NIGHTS),
            "use_validation": USE_VALIDATION,
            "val_night": VAL_NIGHT if USE_VALIDATION else "None",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "class_weight_normal": CLASS_WEIGHT_NORMAL,
            "class_weight_apnea": CLASS_WEIGHT_APNEA,
            "pu_discount": PU_DISCOUNT
        })

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training Multi-Night for {TARGET_TYPE} on {device}...")
        
        print("\n--- Preparing Training Set ---")
        train_dataset = MultiNightApneaDataset(TRAIN_NIGHTS, TARGET_TYPE)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        if USE_VALIDATION:
            print("\n--- Preparing Validation Set ---")
            val_dataset = MultiNightApneaDataset([VAL_NIGHT], TARGET_TYPE)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        else:
            print("\n--- Validation/Early Stopping is DISABLED ---")
        
        model = ConvLSTM().to(device)
        class_weights = torch.tensor([CLASS_WEIGHT_NORMAL, CLASS_WEIGHT_APNEA], dtype=torch.float32).to(device)
        criterion = SimulatedPULoss(class_weights=class_weights, pu_discount=PU_DISCOUNT)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_weights = None
        
        print(f"\nStarting Training (Max Epochs: {MAX_EPOCHS})...")
        
        for epoch in range(MAX_EPOCHS):
            model.train() 
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_x.to(device))
                ce_loss = criterion(predictions, batch_y.to(device))
                
                probs = torch.softmax(predictions, dim=1) 
                apnea_probs = probs[:, 1, :] 
                flicker_penalty = torch.mean(torch.abs(apnea_probs[:, 1:] - apnea_probs[:, :-1]))
                
                loss = ce_loss + (0.5 * flicker_penalty)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            
            if USE_VALIDATION:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        predictions = model(batch_x.to(device))
                        ce_loss = criterion(predictions, batch_y.to(device))
                        probs = torch.softmax(predictions, dim=1)
                        apnea_probs = probs[:, 1, :]
                        flicker_penalty = torch.mean(torch.abs(apnea_probs[:, 1:] - apnea_probs[:, :-1]))
                        loss = ce_loss + (0.5 * flicker_penalty)
                        val_loss += loss.item()
                        
                avg_val_loss = val_loss / len(val_loader)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print(f"   -> [New Best State Saved]")
                else:
                    epochs_without_improvement += 1
                    print(f"   -> [No Improvement: {epochs_without_improvement}]")
                    
                    if epochs_without_improvement >= PATIENCE:
                        print(f"\n🛑 Early Stopping Triggered.")
                        break
            else:
                print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | (No Validation)")
                best_model_weights = copy.deepcopy(model.state_dict())

        # Save Final Best Weights
        save_name = f'penta_lstm_{TARGET_TYPE}_weights.pth'
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights) 
        torch.save(model.state_dict(), save_name)
        
        # --- 2. LOG ARTIFACTS TO MLFLOW ---
        mlflow.log_artifact(save_name)
        print(f"\n✅ Training Complete. Best weights saved to {save_name} and logged to MLflow!")

        # --- 3. AUTOMATED FINAL EVALUATION ---
        if len(TEST_NIGHTS) > 0:
            print("\n==================================================")
            print("🚀 RUNNING AUTOMATED EVALUATION ON TEST NIGHTS 🚀")
            print("==================================================")
            
            # Local import to prevent circular dependency with calculate_clinical_metrics_sft.py
            from calculate_clinical_metrics_sft import evaluate_full_night
            
            for test_night in TEST_NIGHTS:
                print(f"\nEvaluating Night {test_night}...")
                results = evaluate_full_night(model, test_night, TARGET_TYPE, device)
                
                # Prefix metrics so they don't overwrite each other in MLflow (e.g., Night5_f1_score)
                mlflow_metrics = {}
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                        mlflow_metrics[f"Night{test_night}_{key}"] = value
                    else:
                        print(f"  {key}: {value}")
                        mlflow_metrics[f"Night{test_night}_{key}"] = float(value) # MLflow requires floats
                
                # Log to MLflow
                mlflow.log_metrics(mlflow_metrics)
            
            print("\n✅ Evaluation metrics successfully logged to MLflow!")

if __name__ == "__main__": 
    train_model()