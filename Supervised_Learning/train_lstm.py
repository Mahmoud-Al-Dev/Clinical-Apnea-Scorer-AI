import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =================================================================
# --- USER CONTROLS ---
# =================================================================
# Change this to 'OSA' or 'CA' to train the specific model!
TARGET_TYPE = 'OSA' 
NIGHT_TO_TEST = 1
# =================================================================

class ApneaDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.x = torch.tensor(np.load(x_file), dtype=torch.float32)
        self.y = torch.tensor(np.load(y_file), dtype=torch.long).squeeze(-1)
        
        # Mapping the 7-channel array to the 5 AI channels
        self.ai_indices = [0, 3, 4, 5, 6, 7]
        
    def __len__(self): return len(self.x)
    
    def __getitem__(self, idx): 
        # ONLY return the 5 AI channels!
        return self.x[idx, :, self.ai_indices], self.y[idx]

class PentaLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(PentaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True 
        )
        
        # CHANGED: Output 2 classes (0=Normal Breathing, 1=Apnea Event)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out) 
        
        # PyTorch CrossEntropy expects shape: (Batch, Classes, Timesteps)
        predictions = predictions.permute(0, 2, 1)
        
        return predictions

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Binary SFT for {TARGET_TYPE} on {device}...")
    
    y_filename = f'Y_{TARGET_TYPE}_{NIGHT_TO_TEST}.npy'
    dataset = ApneaDataset(f'X_{NIGHT_TO_TEST}.npy', y_filename)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PentaLSTM().to(device)
    
    # 1. Lowered the panic weight. 5.0 is plenty to make it care about OSA.
    class_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 2. Bumped LR slightly to help it find the path
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            predictions = model(batch_x.to(device))
            
            # 3. Pure, unadulterated Cross Entropy. No flicker penalty.
            loss = criterion(predictions, batch_y.to(device))

            loss.backward()
            
            # 4. THE MAGIC FIX: Gradient Clipping! 
            # This stops the LSTM from overshooting and bouncing on noisy OSA data.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    save_name = f'penta_lstm_{TARGET_TYPE}_weights.pth'
    torch.save(model.state_dict(), save_name)
    print(f"✅ Saved weights to {save_name}")
if __name__ == "__main__": train_model()