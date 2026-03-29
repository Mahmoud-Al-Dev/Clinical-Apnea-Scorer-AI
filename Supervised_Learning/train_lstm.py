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
        
        # Mapping the 7-channel array to the 6 AI channels
        self.ai_indices = [0, 3, 4, 5, 6, 7]
        
    def __len__(self): return len(self.x)
    
    def __getitem__(self, idx): 
        # ONLY return the 5 AI channels!
        return self.x[idx, :, self.ai_indices], self.y[idx]

class ConvLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(ConvLSTM, self).__init__()
        
        # 1. The CNN Front-End (The "Eyes")
        # Takes your 6 raw channels and learns 32 complex spatial features
        self.cnn = nn.Sequential(
            # kernel_size=7 looks at ~0.2 seconds of data at once (at 32Hz)
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), # Batch normalization helps it train incredibly fast
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # 2. The LSTM Back-End (The "Memory")
        # Notice the input_size is now 32 (fed by the CNN), not 6!
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True 
        )
        
        # 3. The Classification Head
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        # x comes in as (Batch, Timesteps, Channels) -> e.g., (32, 960, 6)
        
        # PyTorch Conv1d expects (Batch, Channels, Timesteps), so we flip it
        x = x.permute(0, 2, 1) 
        
        # Extract features with CNN
        cnn_features = self.cnn(x)
        
        # Flip it back for the LSTM -> (Batch, Timesteps, Features)
        cnn_features = cnn_features.permute(0, 2, 1)
        
        # Pass through the rest of your normal architecture
        lstm_out, _ = self.lstm(cnn_features)
        predictions = self.fc(lstm_out) 
        
        # (Batch, Classes, Timesteps) for CrossEntropyLoss
        predictions = predictions.permute(0, 2, 1)
        
        return predictions

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Binary SFT for {TARGET_TYPE} on {device}...")
    
    y_filename = f'Y_{TARGET_TYPE}_{NIGHT_TO_TEST}.npy'
    dataset = ApneaDataset(f'X_{NIGHT_TO_TEST}.npy', y_filename)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ConvLSTM().to(device)
    
    class_weights = torch.tensor([1.0, 20], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 2. Bumped LR slightly to help it find the path
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            predictions = model(batch_x.to(device))
            
            # 1. Standard Cross Entropy Loss
            ce_loss = criterion(predictions, batch_y.to(device))
            
            # 2. THE FLICKER PENALTY (Temporal Continuity Loss)
            # Convert logits to probabilities
            probs = torch.softmax(predictions, dim=1) 
            # Isolate the probabilities for the 'Apnea' class (index 1)
            apnea_probs = probs[:, 1, :] 
            
            # Calculate the absolute difference between adjacent timesteps
            flicker_penalty = torch.mean(torch.abs(apnea_probs[:, 1:] - apnea_probs[:, :-1]))
            
            # 3. Combine them (lambda = 0.5 is a good starting point)
            loss = ce_loss + (0.5 * flicker_penalty)

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