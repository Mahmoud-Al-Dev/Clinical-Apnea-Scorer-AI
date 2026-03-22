import gymnasium as gym
import numpy as np
from gymnasium import spaces

NIGHT_TO_TEST = 2

class ApneaEnv(gym.Env):
    def __init__(self, target_type='CA', x_path=f'X_{NIGHT_TO_TEST}.npy', 
                 reward_normal=1.0, reward_apnea=15.0, penalty_miss=15.0, penalty_fa=15.0): # <-- ADDED VARIABLES
        super(ApneaEnv, self).__init__()
        
        print(f"Initializing Balanced Apnea Simulator for {target_type}...")
        self.X = np.load(x_path)
        
        y_path = f'Y_{target_type}_{NIGHT_TO_TEST}.npy'
        self.Y = np.load(y_path)
        
        # Store the reward variables
        self.reward_normal = reward_normal
        self.reward_apnea = reward_apnea
        self.penalty_miss = penalty_miss
        self.penalty_fa = penalty_fa
        
        self.num_segments = len(self.X)
        self.current_step = 0
        
        self.apnea_indices = np.where(np.sum(self.Y, axis=(1, 2)) > 0)[0]
        print(f"--> Found {len(self.apnea_indices)} segments containing {target_type} events out of {self.num_segments} total.")
        
        num_channels = self.X.shape[-1]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(960, num_channels), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2] * 960)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if np.random.rand() < 0.2 and len(self.apnea_indices) > 0:
            self.current_step = np.random.choice(self.apnea_indices)
        else:
            self.current_step = np.random.randint(0, self.num_segments)
        
        obs = self.X[self.current_step].astype(np.float32)
        info = {"target": self.Y[self.current_step].flatten()}
        
        self.episode_step_count = 0 
        return obs, info
        
    def step(self, action):
        target = self.Y[self.current_step].flatten()
        
        correct_normal = np.sum((action == target) & (target == 0)) 
        correct_apnea = np.sum((action == target) & (target == 1))
        missed_apnea = np.sum((action == 0) & (target == 1))
        false_alarm = np.sum((action == 1) & (target == 0))
        
        # CHANGED: Use the dynamic variables instead of hardcoded numbers!
        step_reward = float((correct_normal * self.reward_normal) + 
                            (correct_apnea * self.reward_apnea) - 
                            (missed_apnea * self.penalty_miss) - 
                            (false_alarm * self.penalty_fa))
            
        step_reward = step_reward / 960.0
        
        terminated = True 
        truncated = False
        
        num_channels = self.X.shape[-1]
        next_obs = np.zeros((960, num_channels), dtype=np.float32)
        info = {"target": target}
            
        return next_obs, step_reward, terminated, truncated, info