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
TARGET_TYPE = 'CA'  # Change to 'OSA' to train the Obstructive model!
MAX_QUESTIONS_PER_EPOCH = 5 
WARM_UP_EPOCHS = 2 
INPUT_CHANNELS = 6  
# ==========================================

def train_ppo_rlhf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Binary RLHF PPO for {TARGET_TYPE} on Device: {device}")
    
    segment_times = np.load('segment_times.npy') 
    
    # Pass the TARGET_TYPE to the environment
    env = ApneaEnv(target_type=TARGET_TYPE)
    
    agent = ActorCriticLSTM(input_size=INPUT_CHANNELS, hidden_size=128, num_layers=2).to(device)
    
    # Dynamically load the correct SFT weights
    sft_weights_path = f'penta_lstm_{TARGET_TYPE}_weights.pth'
    agent = load_pretrained_supervised_weights(agent, weights_path=sft_weights_path, device=device)
    
    optimizer = optim.Adam(agent.parameters(), lr=0.0003)
    epochs = 5
    
    for epoch in range(epochs):
        questions_asked_this_epoch = 0 
        total_epoch_reward = 0
        
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")
        
        for step in range(500):
            obs, info = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            action_logits, state_value = agent(obs)
            probs = torch.softmax(action_logits, dim=-1)
            
            m = Categorical(probs)
            action_tensor = m.sample() 
            log_prob = m.log_prob(action_tensor).mean() 
            action_numpy = action_tensor[0].cpu().numpy().astype(int)
            
            with torch.no_grad():
                greedy_action_tensor = torch.argmax(probs, dim=-1)
                plot_action_numpy = greedy_action_tensor[0].cpu().numpy().astype(int)
            
            # --- B. HUMAN FEEDBACK ---
            max_probs, _ = torch.max(probs, dim=-1)
            mean_confidence = max_probs.mean().item()
            human_bonus = 0.0 
            
            # Keep max questions consistent throughout training
            current_max_questions = MAX_QUESTIONS_PER_EPOCH

            ask_for_help = False
            
            if questions_asked_this_epoch < current_max_questions:
                # 1. Did the AI actually draw red lines?
                ai_found_apnea = np.sum(plot_action_numpy == 1) > 30
                
                # 2. THE OLD SYSTEM: Only ask if it found an event AND isn't 100% sure
                if ai_found_apnea and mean_confidence < 0.99: 
                    ask_for_help = True

            if ask_for_help:
                current_seg_idx = env.current_step
                real_start_time = segment_times[current_seg_idx, 0]
                real_end_time = segment_times[current_seg_idx, -1]
                time_axis = segment_times[current_seg_idx]
                
                print(f"\n⚠️ [ACTIVE LEARNING {questions_asked_this_epoch+1}/{current_max_questions}]")
                print(f"AI Confidence ({mean_confidence:.2f}): Checking {real_start_time:.1f}s to {real_end_time:.1f}s.")
                
                # Dynamic feature names based on channel count
                feature_names = ['PFlow', 'Abdomen','Ratio', 'SaO2_Deriv', 'PFlow_Var', 'Vitalog2']
                if INPUT_CHANNELS == 7: feature_names.append('Heart_Rate')
                
                fig, axes = plt.subplots(INPUT_CHANNELS, 1, figsize=(12, 2 * INPUT_CHANNELS), sharex=True)
                plot_color = 'red' if TARGET_TYPE == 'CA' else 'orange'
                
                for i, ax in enumerate(axes):
                    sig = obs[0, :, i].cpu().numpy()
                    ax.plot(time_axis, sig, color='blue', alpha=0.7)
                    
                    # Highlight the AI's binary guess
                    ax.fill_between(time_axis, np.min(sig), np.max(sig), where=(plot_action_numpy == 1), color=plot_color, alpha=0.3, label=f'AI Pred: {TARGET_TYPE}' if i==0 else "")
                    
                    ax.set_ylabel(feature_names[i])
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.legend(loc='upper right')
                        ax.set_title(f"Clinical Review: Segment {current_seg_idx}")

                plt.xlabel("Real Elapsed Time (Seconds)")
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1) 
                
                # CHANGED: Balanced Context-Aware Feedback
                # Because it ONLY asks questions when ai_found_apnea is True, the math is simple!
                user_input = input(f"Classify {TARGET_TYPE}: [1] True Event, [0] False Alarm, [s] Skip: ").strip()
                plt.close()
                
                if user_input == '1':
                    human_bonus = 5.0  # The AI guessed right! Reward it.
                    print(f"--> Human confirmed {TARGET_TYPE}! Good job AI.")
                elif user_input == '0':
                    human_bonus = -5.0 # The AI guessed wrong (False Alarm)! Punish it.
                    print("--> Human says FALSE ALARM. Penalty applied.")
                else:
                    print("--> Skipped.")
                    
                questions_asked_this_epoch += 1

            # --- C. ENVIRONMENT REWARD & OPTIMIZATION ---
            next_obs, step_reward, terminated, truncated, info = env.step(action_numpy)
            
            final_reward = step_reward + human_bonus
            total_epoch_reward += final_reward
            
            final_reward_tensor = torch.tensor([final_reward], dtype=torch.float32).to(device)
            pred_value = state_value.view(-1) 
            
            advantage = final_reward_tensor - pred_value.detach()
            actor_loss = -(log_prob * advantage) 
            critic_loss = nn.MSELoss()(pred_value, final_reward_tensor)
            
            # CHANGED: Calculate Entropy to force the AI to keep an open mind
            entropy = m.entropy().mean()
            
            # CHANGED: Subtract a small entropy bonus (0.01) from the final loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"Epoch {epoch + 1} Completed. Total Reward: {total_epoch_reward:.2f}")

    # Save dedicated RLHF weights
    save_path = f'rlhf_penta_lstm_{TARGET_TYPE}_weights.pth'
    torch.save(agent.state_dict(), save_path)
    print(f"\n✅ Binary RLHF Training Complete! Saved as '{save_path}'")

if __name__ == "__main__":
    train_ppo_rlhf()