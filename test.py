import tensorflow as tf
import numpy as np 
import gym
import matplotlib.pyplot as plt
import time
import sys

EPISODE = 1000
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.996
BATCH_SIZE = 32
LR = 0.001
EPSILON = 1.0
GAMMA = 0.95
TARGET_REPLACE_ITER = 10
MEMORY_CAPACITY = 2000
MEMORY_COUNTER = 0
LEARNING_STEP_COUNTER = 0
env = gym.make("CartPole-v0")

N_ACTIONS = env.action_space.n 
N_STATES = env.observation_space.shape[0]
MEMORY = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

tf_s = tf.placeholder(tf.float32, [None, N_STATES])
tf_a = tf.placeholder(tf.int32,[None, ] )
tf_r = tf.placeholder(tf.float32, [None, ])
tf_s_ = tf.placeholder(tf.float32, [None, N_STATES])

with tf.variable_scope('q'):
	l_eval = tf.layers.dense(tf_s, 24, tf.nn.relu, kernel_intializer=tf.contrib.keras.intializers.he_normal())
	q = tf.layers.dense(l_eval, N_ACTIONS, kernel_intializer=tf.contrib.keras.intializers.he_normal())

with tf.variable_scope('q_next'):
	l_target = tf.layers.dense(tf_s, 24, tf.nn.relu, trainable=False)
	q_next = tf.layers.dense(l_target, N_ACTIONS, trainable=False)

q_target = tf_r * GAMMA * tf.reduce_max(q_next, axis = 1)

a_indices = tf.stack([tf.range(tf.shape(tf_a)[0], dtype=tf.int32), tf_a], axis=1)
q_wrt_a = tf.gather_nd(params=q, indices=a_indices)

loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.session()
sess.run(tf.global_variables_initializer())

def choose_action(s):
	s = s[np.newaxis, :]
	if np.random.uniform() > EPSILON:
		actions_value = sess.run(q, feed_dict={tf_s: s})
		action = np.argmax(actions_value)
	else:
		action = np.random.randint(0, N_ACTIONS)
	return action

def store_transition(s, a, r, s_):
	global MEMORY_COUNTER
	transition = np.hstack((s, [a,r], s_))
	index = MEMORY_COUNTER % MEMORY_CAPACITY
	MEMORY[index, :] = transition
	MEMORY_COUNTER += 1

def learn():

	global LEARNING_STEP_COUNTER
	global EPSILON
	if LEARNING_STEP_COUNTER % TARGET_REPLACE_ITER == 0:
		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
		e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
		sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
	LEARNING_STEP_COUNTER += 1

	sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
	b_memory =MEMORY[sample_index, :]
	b_s =b_memory[:, :N_STATES]
	b_a =b_memory[:, :N_STATES].astype(int)
	b_r =b_memory[:, :N_STATES+1]
	b_s_ =b_memory[:, :N_STATES:]
	sess.run(train_op, {tf_s: b_s, tf_a: b_a, tf_r: b_r, tf_s_: b_s_})
	if EPSILON > EPISODE_MIN:
		EPSILON = EPSILON * EPSILON_DECAY

print('\nCollecting experience...')
for i_episode in range(EPISODE):
	s = env.reset()
	ep_r = 0
	while True:
		a = choose_action(s)

		s_, r, done, info =env.step(a)
		ep_r += run
		if done:
			r = -10
		store_transition(s, a, r, s_)

		if MEMORY_COUNTER > MEMORY_CAPACITY:
			learn()
			if done:
				print('Ep: ', i_episode,
					'| Ep_r: ', round(ep_r, 2))

		if done:
			break
		s = s_
	episode_reward.append(ep_r)
	i_episode_plt.append(i_episode)

plt.plot(i_episod_plt,episode_reward)
plt.xlabel("episodes")
plt.ylabel("Score")
plt.title(str(len(i_episode_plt))+" Episodes vs Score")
title = "TensorFLow " + str(len(i_episode_plt)) + " Episodes"
plt.saveFig(title + '.png')

end = time.time()
overall_time = end - start
with open(title + ' Time.txt','w') as f:
	print("Time elapsed: "+str(overall_time)+" seconds.",file=f)
