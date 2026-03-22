import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical 
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sleep_Apnea_RLHF_Night_2")

from apnea_env import ApneaEnv
from actor_critic_lstm import ActorCriticLSTM, load_pretrained_supervised_weights
from calculate_clinical_metrics import evaluate_full_night # <-- IMPORTED EVALUATOR

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TYPE = 'CA'
NIGHT_TO_TEST = 2
MAX_QUESTIONS_PER_EPOCH = 8
WARM_UP_EPOCHS = 2 
INPUT_CHANNELS = 6  
# ==========================================

def train_ppo_rlhf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Binary RLHF PPO for {TARGET_TYPE} on Device: {device}")
    
    # --- HYPERPARAMETERS TO TRACK ---
    LEARNING_RATE = 0.0001
    EPOCHS = 5
    STEPS_PER_EPOCH = 100
    CONFIDENCE_THRESHOLD = 0.91
    HUMAN_REWARD_BONUS = 10.0
    HUMAN_PENALTY_MINUS = -5.0
    CRITIC_LOSS_COEF = 0.5
    ENTROPY_LOSS_COEF = 0.01
    FEATURE_NAMES = "['PFlow_Clean', 'Abdomen_Clean', 'Ratio', 'SaO2_Deriv', 'PFlow_Var', 'Vitalog2']"
    
    # Environment Rewards Setup (Based on Target)
    ENV_REWARD_APNEA = 15.0 if TARGET_TYPE == 'CA' else 20.0
    ENV_PENALTY_MISS = 15.0
    ENV_PENALTY_FA = 15.0 if TARGET_TYPE == 'CA' else 50.0

    # Pass the tracked rewards to the environment
    env = ApneaEnv(target_type=TARGET_TYPE, 
                   reward_apnea=ENV_REWARD_APNEA, 
                   penalty_miss=ENV_PENALTY_MISS, 
                   penalty_fa=ENV_PENALTY_FA)
    
    agent = ActorCriticLSTM(input_size=INPUT_CHANNELS, hidden_size=128, num_layers=2).to(device)
    sft_weights_path = f'penta_lstm_{TARGET_TYPE}_weights.pth'
    agent = load_pretrained_supervised_weights(agent, weights_path=sft_weights_path, device=device)
    
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy') 

    # =======================================================
    # START MLFLOW RUN
    # =======================================================
    with mlflow.start_run(run_name=f"RLHF_{TARGET_TYPE}_Night_{NIGHT_TO_TEST}"):
        
        # 1. Log all our parameters up front
        mlflow.log_params({
            "target_type": TARGET_TYPE,
            "night_to_test": NIGHT_TO_TEST,
            "max_questions": MAX_QUESTIONS_PER_EPOCH,
            "warm_up_epochs": WARM_UP_EPOCHS,
            "input_channels": INPUT_CHANNELS,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "human_reward_bonus": HUMAN_REWARD_BONUS,
            "human_penalty_minus": HUMAN_PENALTY_MINUS,
            "critic_loss_coef": CRITIC_LOSS_COEF,
            "entropy_loss_coef": ENTROPY_LOSS_COEF,
            "feature_names": FEATURE_NAMES,
            "env_reward_apnea": ENV_REWARD_APNEA,
            "env_penalty_miss": ENV_PENALTY_MISS,
            "env_penalty_fa": ENV_PENALTY_FA
        })
        
        global_step = 0 # To track high-resolution metrics
        
        for epoch in range(EPOCHS):
            questions_asked_this_epoch = 0 
            total_epoch_reward = 0
            
            print(f"\n--- Starting Epoch {epoch + 1}/{EPOCHS} ---")
            
            for step in range(STEPS_PER_EPOCH):
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
                
                max_probs, _ = torch.max(probs, dim=-1)
                mean_confidence = max_probs.mean().item()
                human_bonus = 0.0 

                ask_for_help = False
                
                if questions_asked_this_epoch < MAX_QUESTIONS_PER_EPOCH:
                    ai_found_apnea = np.sum(plot_action_numpy == 1) > 30
                    if ai_found_apnea and mean_confidence < CONFIDENCE_THRESHOLD: 
                        ask_for_help = True

                if ask_for_help:
                    current_seg_idx = env.current_step
                    real_start_time = segment_times[current_seg_idx, 0]
                    real_end_time = segment_times[current_seg_idx, -1]
                    time_axis = segment_times[current_seg_idx]

                    print(f"\n⚠️ [ACTIVE LEARNING {questions_asked_this_epoch+1}/{MAX_QUESTIONS_PER_EPOCH}]")
                    print(f"AI Confidence ({mean_confidence:.2f}): Checking {real_start_time:.1f}s to {real_end_time:.1f}s.")

                    feature_names = ['PFlow_Clean', 'Abdomen_Clean', 'Ratio', 'SaO2_Deriv', 'PFlow_Var', 'Vitalog2']
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
                    
                    user_input = input(f"Classify {TARGET_TYPE}: [1] True Event, [0] False Alarm, [s] Skip: ").strip()

                    plt.close(fig) 
                    
                    if user_input == '1':
                        human_bonus = HUMAN_REWARD_BONUS
                        print(f"--> Human confirmed {TARGET_TYPE}! Good job AI.")
                        mlflow.log_metric("human_interventions_correct", 1, step=global_step)
                    elif user_input == '0':
                        human_bonus = HUMAN_PENALTY_MINUS
                        print("--> Human says FALSE ALARM. Penalty applied.")
                        mlflow.log_metric("human_interventions_false_alarm", 1, step=global_step)
                        
                    questions_asked_this_epoch += 1

                # Step the environment
                next_obs, step_reward, terminated, truncated, info = env.step(action_numpy)
                final_reward = step_reward + human_bonus
                total_epoch_reward += final_reward
                
                # --- LOG HIGH RESOLUTION METRICS ---
                mlflow.log_metric("step_reward", step_reward, step=global_step)
                mlflow.log_metric("human_bonus", human_bonus, step=global_step)
                mlflow.log_metric("total_step_reward", final_reward, step=global_step)
                mlflow.log_metric("mean_confidence", mean_confidence, step=global_step)
                
                # Loss Math
                final_reward_tensor = torch.tensor([final_reward], dtype=torch.float32).to(device)
                pred_value = state_value.view(-1) 
                
                advantage = final_reward_tensor - pred_value.detach()
                actor_loss = -(log_prob * advantage) 
                critic_loss = nn.MSELoss()(pred_value, final_reward_tensor)
                entropy = m.entropy().mean()
                
                loss = actor_loss + (CRITIC_LOSS_COEF * critic_loss) - (ENTROPY_LOSS_COEF * entropy)
                
                # Log Losses
                mlflow.log_metric("actor_loss", actor_loss.item(), step=global_step)
                mlflow.log_metric("critic_loss", critic_loss.item(), step=global_step)
                mlflow.log_metric("entropy", entropy.item(), step=global_step)
                mlflow.log_metric("total_loss", loss.item(), step=global_step)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                
                global_step += 1

            print(f"Epoch {epoch + 1} Completed. Total Reward: {total_epoch_reward:.2f}")
            mlflow.log_metric("epoch_total_reward", total_epoch_reward, step=epoch)

        # =======================================================
        # END OF TRAINING - AUTOMATIC CLINICAL EVALUATION
        # =======================================================
        save_path = f'rlhf_penta_lstm_{TARGET_TYPE}_weights.pth'
        torch.save(agent.state_dict(), save_path)
        print(f"\n✅ Binary RLHF Training Complete! Saved as '{save_path}'")
        
        print("\n📊 Running Automatic Clinical Evaluation on Full Night...")
        clinical_results = evaluate_full_night(agent, NIGHT_TO_TEST, TARGET_TYPE, device)
        
        print("Logging Clinical Metrics to MLflow...")
        for metric_name, value in clinical_results.items():
            mlflow.log_metric(f"final_{metric_name}", value)
            
        # Save the model artifact to MLflow
        mlflow.pytorch.log_model(agent, "model_weights")
        print("✅ Run complete. All data sent to MLflow Dashboard!")

if __name__ == "__main__":
    train_ppo_rlhf()