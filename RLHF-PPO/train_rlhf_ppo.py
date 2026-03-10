import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical 
import numpy as np
import matplotlib.pyplot as plt

from apnea_env import ApneaEnv
from actor_critic_lstm import ActorCriticLSTM, load_pretrained_supervised_weights

# ==========================================
# --- USER CONTROLS ---
# ==========================================
MAX_QUESTIONS_PER_EPOCH = 5 
WARM_UP_EPOCHS = 2 # During these epochs, the AI will ask questions regardless of confidence
# ==========================================

def train_ppo_rlhf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Multi-Class RLHF PPO on Device: {device}")
    
    segment_times = np.load('segment_times.npy') 
    
    env = ApneaEnv()
    agent = ActorCriticLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
    agent = load_pretrained_supervised_weights(agent, device=device)
    
    optimizer = optim.Adam(agent.parameters(), lr=0.0003)
    epochs = 5
    
    for epoch in range(epochs):
        questions_asked_this_epoch = 0 
        total_epoch_reward = 0
        
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")
        
        # Since 1 segment = 1 episode, we loop through 500 independent segments per epoch
        for step in range(500):
            obs, info = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            action_logits, state_value = agent(obs)
            probs = torch.softmax(action_logits, dim=-1)
            
            # --- THE RL SAMPLING ---
            m = Categorical(probs)
            action_tensor = m.sample() 
            log_prob = m.log_prob(action_tensor).sum() 
            action_numpy = action_tensor[0].cpu().numpy().astype(int)
            
            # --- TIP 2: GREEDY DECODING FOR EVAL ---
            # We use the absolute best guess (no sampling) for the UI plot
            with torch.no_grad():
                greedy_action_tensor = torch.argmax(probs, dim=-1)
                plot_action_numpy = greedy_action_tensor[0].cpu().numpy().astype(int)
            
            # --- B. HUMAN FEEDBACK ---
            max_probs, _ = torch.max(probs, dim=-1)
            mean_confidence = max_probs.mean().item()
            human_bonus = 0.0 
            
            if epoch < WARM_UP_EPOCHS:
                current_max_questions = 5
            else:
                current_max_questions = 3

            ask_for_help = False
            
            if questions_asked_this_epoch < current_max_questions:
                
                # Check if the AI is actively trying to draw an Apnea box (CA or OSA)
                ai_found_apnea = np.sum(plot_action_numpy > 0) > 30
                
                if epoch < WARM_UP_EPOCHS:
                    # WARM-UP: If it finds an apnea, ALWAYS ask the human to verify 
                    # This builds a strong baseline for distinguishing CA vs OSA
                    if ai_found_apnea:
                        ask_for_help = True
                else:
                    # POST-WARM-UP: Only ask if it finds an apnea AND is mathematically unsure
                    if ai_found_apnea and mean_confidence < 0.95:
                        ask_for_help = True

            if ask_for_help:
                current_seg_idx = env.current_step
                real_start_time = segment_times[current_seg_idx, 0]
                real_end_time = segment_times[current_seg_idx, -1]
                time_axis = segment_times[current_seg_idx]
                
                print(f"\n⚠️ [ACTIVE LEARNING {questions_asked_this_epoch+1}/{current_max_questions}]")
                if epoch < WARM_UP_EPOCHS:
                    print(f"Warm-Up Phase: Forced check between {real_start_time:.1f}s and {real_end_time:.1f}s.")
                else:
                    print(f"Low Confidence ({mean_confidence:.2f}): Checking {real_start_time:.1f}s to {real_end_time:.1f}s.")
                
                feature_names = ['PFlow', 'Abdomen','Ratio', 'SaO2_Deriv', 'PFlow_Var', 'Vitalog2']
                fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
                
                for i, ax in enumerate(axes):
                    sig = obs[0, :, i].cpu().numpy()
                    ax.plot(time_axis, sig, color='blue', alpha=0.7)
                    
                    # TIP 2 applied: We plot the GREEDY array so the UI isn't chaotic
                    ax.fill_between(time_axis, np.min(sig), np.max(sig), where=(plot_action_numpy == 1), color='red', alpha=0.3, label='AI Pred: CA' if i==0 else "")
                    ax.fill_between(time_axis, np.min(sig), np.max(sig), where=(plot_action_numpy == 2), color='orange', alpha=0.4, label='AI Pred: OSA' if i==0 else "")
                    
                    ax.set_ylabel(feature_names[i])
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.legend(loc='upper right')
                        ax.set_title(f"Clinical Review: Segment {current_seg_idx}")

                plt.xlabel("Real Elapsed Time (Seconds)")
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1) 
                
                user_input = input("Classify: [1] CA, [2] OSA, [0] False Alarm, [s] Skip: ").strip()
                plt.close()
                
                # TIP 4: Scale down the human bonus!
                if user_input == '1':
                    human_bonus = 2.0  
                    print("--> Human says CENTRAL.")
                elif user_input == '2':
                    human_bonus = 2.0  
                    print("--> Human says OBSTRUCTIVE.")
                elif user_input == '0':
                    human_bonus = -1.0 
                    print("--> Human says FALSE ALARM.")
                else:
                    print("--> Skipped.")
                    
                questions_asked_this_epoch += 1

            # --- C. ENVIRONMENT REWARD & OPTIMIZATION ---
            # Environment steps using the sampled array (action_numpy) to keep RL math correct
            next_obs, step_reward, terminated, truncated, info = env.step(action_numpy)
            
            # TIP 4: Combine and CLIP the reward to prevent exploding gradients
            final_reward = step_reward + human_bonus
            total_epoch_reward += final_reward
            
            final_reward_tensor = torch.tensor([final_reward], dtype=torch.float32).to(device)
            pred_value = state_value.view(-1) 
            
            # --- D. PPO OPTIMIZATION ---
            advantage = final_reward_tensor - pred_value.detach()
            actor_loss = -(log_prob * advantage) 
            critic_loss = nn.MSELoss()(pred_value, final_reward_tensor)
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # FIX 2: THE SAFETY NET. Add this line right before optimizer.step()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            
            optimizer.step()

        print(f"Epoch {epoch + 1} Completed. Total Reward: {total_epoch_reward:.2f}")

    torch.save(agent.state_dict(), 'rlhf_penta_lstm_weights.pth')
    print("\n✅ Multi-Class RLHF Training Complete! Saved as 'rlhf_penta_lstm_weights.pth'")

if __name__ == "__main__":
    train_ppo_rlhf()