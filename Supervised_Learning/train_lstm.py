import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =================================================================
# --- USER CONTROLS ---
# =================================================================
# Change this to 'OSA' or 'CA' to train the specific model!
TARGET_TYPE = 'CA' 
# =================================================================

class ApneaDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.x = torch.tensor(np.load(x_file), dtype=torch.float32)
        # Flatten the labels to (Batch, 960) and make them Long integers!
        self.y = torch.tensor(np.load(y_file), dtype=torch.long).squeeze(-1)
        
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

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
    
    # Dynamically load the correct label file based on TARGET_TYPE
    y_filename = f'Y_train_Labels_{TARGET_TYPE}.npy'
    dataset = ApneaDataset('X_train_PentaLSTM.npy', y_filename)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PentaLSTM().to(device)
    
    # CHANGED: Only 2 weights now. Class 0 gets weight 1, Class 1 gets weight 35.
    class_weights = torch.tensor([1.0, 80], dtype=torch.float32).to(device)
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

    # Dynamically save the weights so they don't overwrite each other!
    save_name = f'penta_lstm_{TARGET_TYPE}_weights.pth'
    torch.save(model.state_dict(), save_name)
    print(f"✅ Saved weights to {save_name}")

if __name__ == "__main__": train_model()