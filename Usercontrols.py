import gymnasium as gym
from gymnasium.utils.play import play
env = gym.make('ALE/Blackjack-v5', render_mode='rgb_array', full_action_space=True)
play(env, zoom=5)
