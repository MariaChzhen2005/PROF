import gym
from gym import spaces
import numpy as np

class EplusEnvAdapter(gym.Wrapper):
    """Adapter to make EnergyPlus environment compatible with OpenAI Gym interface"""
    
    def __init__(self, env):
        super(EplusEnvAdapter, self).__init__(env)
        # Keep the same spaces as the original environment
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
    def reset(self):
        reset_result = self.env.reset()
        
        # Handle different possible return formats
        if isinstance(reset_result, tuple):
            if len(reset_result) == 3:
                timeStep, obs, isTerminal = reset_result
            else:
                # If reset_result is a tuple but not length 3, 
                # assume it returns observation only
                obs = reset_result[0]
                timeStep = 0  # Default value
                isTerminal = False
        else:
            # If not a tuple, assume it's just the observation
            obs = reset_result
            timeStep = 0  # Default value
            isTerminal = False
            
        return timeStep, obs, isTerminal
        
    def step(self, action):
        step_result = self.env.step(action)
        
        # EnergyPlus environment returns (timestep, obs, isTerminal)
        if isinstance(step_result, tuple) and len(step_result) == 3:
            timeStep, obs, isTerminal = step_result
            # Create the format expected by OpenAI Gym
            reward = 0  # You may want to compute a proper reward
            info = {"timeStep": timeStep}
            return obs, reward, isTerminal, info
        else:
            # If it already follows Gym format
            return step_result