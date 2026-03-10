import torch
import torch.nn as nn
import numpy as np

# 1. Define the model
class PentaLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(PentaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions.permute(0, 2, 1)

# --- THE FIX: Helper function to calculate loss in small bites! ---
def calculate_batched_loss(X_np, Y_np, model, criterion, device, batch_size=64):
    num_samples = len(X_np)
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Only push a tiny batch of 64 to the GPU at a time
            batch_x = torch.tensor(X_np[i : i + batch_size], dtype=torch.float32).to(device)
            batch_y = torch.tensor(Y_np[i : i + batch_size], dtype=torch.long).to(device)
            
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            
            # Keep a running total of the loss
            total_loss += loss.item() * len(batch_x)
            
    return total_loss / num_samples
# ------------------------------------------------------------------

def run_permutation_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Permutation Importance Test on {device}...")

    # 2. Load the 12-feature data and labels (Keep them on the CPU for now!)
    X = np.load('X_train_PentaLSTM.npy')
    Y = np.load('Y_train_Labels.npy').squeeze(-1)

    # 3. Load the trained 12-feature weights
    model = PentaLSTM(input_size=6).to(device)
    model.load_state_dict(torch.load('penta_lstm_weights.pth', map_location=device, weights_only=True))
    model.eval()

    class_weights = torch.tensor([1.0, 20.0, 20.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Calculate Baseline Loss (Using the batched helper)
    baseline_loss = calculate_batched_loss(X, Y, model, criterion, device)
    print(f"Baseline Loss (All 12 Features Intact): {baseline_loss:.4f}\n")

    # 5. The Scramble Test
    feature_names = [
        'PFlow_Clean',  'Abdomen_Clean', 'Ratio', 'SaO2_Deriv', 'PFlow_Var','Vitalog2'
    ]
    
    importances = {}

    for i, name in enumerate(feature_names):
        # Make a fresh copy of the clean data
        X_scrambled = X.copy()
        
        # Completely randomize/shuffle the data for THIS specific feature
        np.random.shuffle(X_scrambled[:, :, i])
        
        # Calculate the new loss using the batched helper!
        scrambled_loss = calculate_batched_loss(X_scrambled, Y, model, criterion, device)
            
        # Importance = How much worse did the model get when this feature was destroyed?
        loss_increase = scrambled_loss - baseline_loss
        importances[name] = loss_increase
        print(f"Scrambled {name}: Loss increased by +{loss_increase:.4f}")

    # 6. Print the Leaderboard
    print("\n🏆 ULTIMATE FEATURE LEADERBOARD 🏆")
    print("(Higher number = More Important to the AI)")
    print("-" * 40)
    sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    for rank, (name, imp) in enumerate(sorted_importances):
        print(f"#{rank+1}. {name}: {imp:.4f}")

if __name__ == "__main__":
    run_permutation_test()