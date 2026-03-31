import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy 

# =================================================================
# --- USER CONTROLS ---
# =================================================================
TARGET_TYPE = 'OSA' 

# 1. List of nights to combine for training
TRAIN_NIGHTS = [1, 3] 

# 2. Single night to use as the validation test
VAL_NIGHT = 1

# 3. Early Stopping Settings
MAX_EPOCHS = 50
PATIENCE = 10 
# =================================================================

class MultiNightApneaDataset(Dataset):
    def __init__(self, nights_list, target_type):
        x_list = []
        y_list = []
        
        for night in nights_list:
            print(f"Loading Night {night} for dataset...")
            
            # Load Silver Standard for Night 3 if available
            if night == 3:
                y_file = f'Y_{target_type}_{night}_SILVER.npy'
                try:
                    y_data = np.load(y_file)
                except FileNotFoundError:
                    print(f"  [!] Silver Standard not found for Night 3. Using original labels.")
                    y_file = f'Y_{target_type}_{night}.npy'
                    y_data = np.load(y_file)
            else:
                y_file = f'Y_{target_type}_{night}.npy'
                y_data = np.load(y_file)
                
            x_file = f'X_{night}.npy'
            x_data = np.load(x_file)
            
            x_list.append(x_data)
            y_list.append(y_data)
            
        self.x = torch.tensor(np.concatenate(x_list, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.long).squeeze(-1)
        
        self.ai_indices = [0, 3, 4, 5, 6, 7]
        
    def __len__(self): return len(self.x)
    
    def __getitem__(self, idx): 
        return self.x[idx, :, self.ai_indices], self.y[idx]

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
            input_size=32, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True 
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

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multi-Night for {TARGET_TYPE} on {device}...")
    
    print("\n--- Preparing Training Set ---")
    train_dataset = MultiNightApneaDataset(TRAIN_NIGHTS, TARGET_TYPE)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("\n--- Preparing Validation Set ---")
    val_dataset = MultiNightApneaDataset([VAL_NIGHT], TARGET_TYPE)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = ConvLSTM().to(device)
    
    class_weights = torch.tensor([1.0, 20], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_weights = None
    
    print(f"\nStarting Training (Max Epochs: {MAX_EPOCHS}, Patience: {PATIENCE})...")
    
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

    save_name = f'penta_lstm_{TARGET_TYPE}_weights.pth'
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights) 
    torch.save(model.state_dict(), save_name)
    
    print(f"\n✅ Training Complete. Best weights saved to {save_name}!")

if __name__ == "__main__": 
    train_model()