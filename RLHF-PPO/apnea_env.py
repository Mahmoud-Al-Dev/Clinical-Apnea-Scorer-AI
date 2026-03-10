import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ApneaEnv(gym.Env):
    # CHANGED: Defaulting to the new Master Dataset
    def __init__(self, x_path='X_train_PentaLSTM.npy', y_path='Y_train_Labels.npy'):
        super(ApneaEnv, self).__init__()
        
        print("Initializing Multi-Class Apnea Environment Simulator...")
        self.X = np.load(x_path)
        self.Y = np.load(y_path)
        
        self.num_segments = len(self.X)
        self.current_step = 0
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(960, 6), dtype=np.float32
        )
        
        # CHANGED: Action Space is now 3 options (0, 1, 2) for all 960 timesteps
        self.action_space = spaces.MultiDiscrete([3] * 960)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # CHANGED: Start at a completely random segment in the Master dataset
        self.current_step = np.random.randint(0, self.num_segments)
        
        obs = self.X[self.current_step].astype(np.float32)
        info = {"target": self.Y[self.current_step].flatten()}
        
        # We also need to track how many steps we've taken in this "episode"
        self.episode_step_count = 0 
        return obs, info
        
    def step(self, action):
        target = self.Y[self.current_step].flatten()
        
        # Count the scenarios
        correct_normal = np.sum((action == target) & (target == 0)) # BROUGHT THIS BACK!
        correct_apnea = np.sum((action == target) & (target > 0))
        missed_apnea = np.sum((action == 0) & (target > 0))
        false_alarm = np.sum((action > 0) & (target == 0))
        wrong_class = np.sum((action > 0) & (target > 0) & (action != target))
        
        # Reward calculation aligned with your 1:35 SFT weights
        step_reward = float((correct_normal * 1.0) + (correct_apnea * 35.0) - (missed_apnea * 35.0) - (false_alarm * 10.0) - (wrong_class * 10.0))
        
        # Normalize the environment reward
        step_reward = step_reward / 960.0
        
        # (Tip 3) ONE SEGMENT = ONE EPISODE. Terminate immediately after 1 step.
        terminated = True 
        truncated = False
        
        # Return empty obs because the episode is over
        next_obs = np.zeros((960, 6), dtype=np.float32)
        info = {"target": target}
            
        return next_obs, step_reward, terminated, truncated, info