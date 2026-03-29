import torch
import torch.nn as nn

class ActorCriticLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(ActorCriticLSTM, self).__init__()
        
        # 1. THE SHARED CNN FRONT-END (Must match your ConvLSTM)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # 2. THE SHARED LSTM (Input is now 32 from CNN)
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True
        )
        
        self.actor_head = nn.Linear(hidden_size * 2, 2)
        self.critic_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (Batch, Timesteps, Channels) -> (Batch, 6, 960)
        x_cnn = x.permute(0, 2, 1) 
        cnn_feats = self.cnn(x_cnn)
        cnn_feats = cnn_feats.permute(0, 2, 1) # (Batch, 960, 32)
        
        lstm_out, _ = self.lstm(cnn_feats)  
        action_logits = self.actor_head(lstm_out)
        
        summary_vector = torch.mean(lstm_out.detach(), dim=1) 
        state_value = self.critic_head(summary_vector) 
        
        return action_logits, state_value

def load_pretrained_supervised_weights(rl_model, weights_path, device='cpu'):
    print(f"Loading ConvLSTM SFT weights from {weights_path}...")
    pretrained_dict = torch.load(weights_path, map_location=device, weights_only=True)
    rl_dict = rl_model.state_dict()
    
    # Map layers: CNN, LSTM, and the Actor Head (from the old 'fc' layer)
    for k, v in pretrained_dict.items():
        if 'cnn' in k or 'lstm' in k:
            rl_dict[k] = v  
        elif 'fc.weight' in k:
            rl_dict['actor_head.weight'] = v  
        elif 'fc.bias' in k:
            rl_dict['actor_head.bias'] = v

    rl_model.load_state_dict(rl_dict)
    print("[SUCCESS] Successfully transplanted CNN, LSTM, and Actor Policy!")
    return rl_model