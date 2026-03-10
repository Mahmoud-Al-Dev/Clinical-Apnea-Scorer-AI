import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =================================================================
# 1. THE DATASET CLASS (The Loader)
# =================================================================
class ApneaDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.x = torch.tensor(np.load(x_file), dtype=torch.float32)
        # Flatten the labels to (Batch, 960) and make them Long integers!
        self.y = torch.tensor(np.load(y_file), dtype=torch.long).squeeze(-1)
        
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
# =================================================================
# 2. THE PENTA-LSTM ARCHITECTURE (The Brain)
# =================================================================
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
        
        # CHANGED: Output 3 classes (0=Normal, 1=CA, 2=OH) instead of 1
        self.fc = nn.Linear(hidden_size * 2, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out) 
        
        # CHANGED: PyTorch CrossEntropy expects shape: (Batch, Classes, Timesteps)
        # We must swap the 'Classes' (dim 2) and 'Timesteps' (dim 1)
        predictions = predictions.permute(0, 2, 1)
        
        return predictions

# =================================================================
# 3. THE TRAINING ENGINE
# =================================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multi-Class SFT on {device}...")
    
    dataset = ApneaDataset('X_train_PentaLSTM.npy', 'Y_train_Labels.npy')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PentaLSTM().to(device)
    
    # Heavily penalize the AI if it misses an apnea!
    class_weights = torch.tensor([1, 35, 35], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 30
    
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x.to(device))
            loss = criterion(predictions, batch_y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'penta_lstm_weights.pth')

if __name__ == "__main__": train_model()