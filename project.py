import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import sys
import keyboard

def run():
	env = gym.make("ALE/Blackjack-v5", render_mode='human', obs_type='rgb',full_action_space=True)
	env.metadata['render_fps']=60

	env.reset()

	num_actions = env.action_space.n
	print(num_actions)

	meaning=env.unwrapped.get_action_meanings()

	actionDict={'w':0, 's':1, 'd':2, 'a':3}

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
		env.metadata['render_fps']=60
		time.sleep(0.05)
	env.close()

if __name__=='__main__':
	run()

env.reset()
env.close()
