import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState
import os

# =================================================================
# --- PROXY DATASET FOR HPO ---
# =================================================================
TARGET_TYPE = 'OSA' 
TRAIN_NIGHTS = [2, 18, 26, 32]  # High-density proxy training set
VAL_NIGHTS = [3, 22]            # Ultra-stable proxy validation set
MAX_EPOCHS = 25                 # Reduced epochs for faster trials
# =================================================================

class MultiNightApneaDataset(Dataset):
    def __init__(self, nights_list, target_type, folder="Nights"): 
        x_list = []
        y_list = []
        for night in nights_list:
            y_file = os.path.join(folder, f'Y_{target_type}_{night}.npy')
            x_file = os.path.join(folder, f'X_{night}.npy')
            x_list.append(np.load(x_file))
            y_list.append(np.load(y_file))
            
        self.x = torch.tensor(np.concatenate(x_list, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.long).squeeze(-1)
        self.ai_indices = [0, 3, 4, 5, 6, 7]
        
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx, :, self.ai_indices], self.y[idx]

class ConvLSTM(nn.Module):
    # (Keep your exact ConvLSTM architecture here as defined in train_lstm.py)
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
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        cnn_features = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_features)
        return self.fc(lstm_out).permute(0, 2, 1)

class SimulatedPULoss(nn.Module):
    def __init__(self, class_weights, pu_discount=0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.pu_discount = pu_discount

    def forward(self, logits, targets):
        base_loss = self.ce(logits, targets) 
        probs = torch.softmax(logits, dim=1)[:, 1, :]
        discount_mask = torch.where((targets == 0) & (probs > 0.5), self.pu_discount, 1.0)
        return (base_loss * discount_mask).mean()

# --- PRE-LOAD DATA IN MEMORY ONCE ---
print("Loading Proxy Datasets into memory...")
train_dataset = MultiNightApneaDataset(TRAIN_NIGHTS, TARGET_TYPE)
val_dataset = MultiNightApneaDataset(VAL_NIGHTS, TARGET_TYPE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # 1. OPTUNA HYPERPARAMETER SEARCH SPACE
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    apnea_weight = trial.suggest_float("apnea_weight", 2.0, 4.5)
    pu_discount = trial.suggest_float("pu_discount", 0.3, 0.85)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # L2 Regularization

    # 2. SETUP DATA & MODEL
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ConvLSTM().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    class_weights = torch.tensor([1.0, apnea_weight], dtype=torch.float32).to(device)
    criterion = SimulatedPULoss(class_weights=class_weights, pu_discount=pu_discount)
    
    # Notice we added weight_decay here to help prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')

    # 3. TRAINING LOOP
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x.to(device))
            ce_loss = criterion(predictions, batch_y.to(device))
            probs = torch.softmax(predictions, dim=1)[:, 1, :] 
            flicker_penalty = torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))
            loss = ce_loss + (0.5 * flicker_penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # 4. VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x.to(device))
                ce_loss = criterion(predictions, batch_y.to(device))
                probs = torch.softmax(predictions, dim=1)[:, 1, :]
                flicker_penalty = torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))
                val_loss += (ce_loss + (0.5 * flicker_penalty)).item()
                
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # 5. OPTUNA PRUNING (The Magic)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

if __name__ == "__main__":
    print(f"🔥 Starting Optuna Hyperparameter Optimization on {device}...")
    
    # We use a MedianPruner to kill trials that are performing worse than the median of past trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    # Run 30 trials
    study.optimize(objective, n_trials=30)

    print("\n==================================================")
    print("🏆 OPTUNA HPO COMPLETE! 🏆")
    print("==================================================")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    print("\n🎯 Best Trial Parameters (Put these in train_lstm.py):")
    trial = study.best_trial
    print(f"  Value (Val Loss): {trial.value}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")