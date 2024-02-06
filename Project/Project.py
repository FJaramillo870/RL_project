import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import sys
import keyboard
#from pynput import keyboard

env = gym.make("ALE/Blackjack-v5", render_mode='human', obs_type='rgb',full_action_space=False)

env.reset()

num_actions = env.action_space.n

meaning=env.unwrapped.get_action_meanings()

actionDict={'w':0, 's':1, 'd':2, 'a':3 }

obs, reward, terminated, truncated, info = env.step(1)
print(info)

plt.figure(figsize=(8,8))
plt.imshow(obs)

totalReward=0
while True:

	event = keyboard.read_event()

	if actionDict.get(event.name, -1) != -1:
		obs, reward, terminated, truncated, info = env.step(actionDict.get(event.name,-1))
	totalReward=totalReward+reward
	env.render()
	time.sleep(0.05)
	#print(event.name)

env.reset()
env.close()