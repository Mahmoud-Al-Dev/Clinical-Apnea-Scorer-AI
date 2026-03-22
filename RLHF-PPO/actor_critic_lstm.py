import torch
import torch.nn as nn

class ActorCriticLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(ActorCriticLSTM, self).__init__()
        
        # THE SHARED BRAIN
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True
        )
        
        # CHANGED: Output 2 classes (0=Normal, 1=Apnea)
        self.actor_head = nn.Linear(hidden_size * 2, 2)
        
        # 2. THE CRITIC HEAD
        self.critic_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        action_logits = self.actor_head(lstm_out)
        
        # The .detach() shield!
        summary_vector = torch.mean(lstm_out.detach(), dim=1) 
        state_value = self.critic_head(summary_vector) 
        
        return action_logits, state_value

# =================================================================
# TRANSFER LEARNING UTILITY
# =================================================================
def load_pretrained_supervised_weights(rl_model, weights_path, device='cpu'):
    print(f"Loading Binary SFT weights from {weights_path}...")
    pretrained_dict = torch.load(weights_path, map_location=device, weights_only=True)
    rl_dict = rl_model.state_dict()
    
    for k, v in pretrained_dict.items():
        if 'lstm' in k:
            rl_dict[k] = v  
        elif 'fc.weight' in k:
            rl_dict['actor_head.weight'] = v  
        elif 'fc.bias' in k:
            rl_dict['actor_head.bias'] = v

    rl_model.load_state_dict(rl_dict)
    print("[SUCCESS] Successfully transplanted BOTH the LSTM memory and Binary Actor Policy!")
    return rl_model