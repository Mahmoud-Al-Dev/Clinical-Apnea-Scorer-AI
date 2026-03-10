import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from actor_critic_lstm import ActorCriticLSTM

print("1. Loading Data...")
X = np.load('X_train_PentaLSTM.npy')
Y_true = np.load('Y_train_Labels.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("2. Loading the Smarter RLHF Agent...")
model = ActorCriticLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
model.load_state_dict(torch.load('rlhf_penta_lstm_weights.pth', map_location=device, weights_only=True))
model.eval()

print("3. Generating Predictions for the entire dataset...")
input_tensor = torch.tensor(X, dtype=torch.float32).to(device)

with torch.no_grad():
    action_logits, _ = model(input_tensor)
    # Convert to probabilities, then to 0s and 1s
    probabilities = torch.sigmoid(action_logits).cpu().numpy()
    predictions = (probabilities > 0.5).astype(int)

print("4. Calculating Final Metrics...")
# We flatten the arrays so we evaluate every single fraction of a second
y_true_flat = Y_true.flatten()
y_pred_flat = predictions.flatten()

acc = accuracy_score(y_true_flat, y_pred_flat)
prec = precision_score(y_true_flat, y_pred_flat)
rec = recall_score(y_true_flat, y_pred_flat)
f1 = f1_score(y_true_flat, y_pred_flat)

print("\n" + "="*50)
print("🏆 RLHF MODEL PERFORMANCE METRICS 🏆")
print("="*50)
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")
print("="*50)
print("\nDetailed Classification Report:")
print(classification_report(y_true_flat, y_pred_flat, target_names=['Normal Breathing', 'Apnea']))