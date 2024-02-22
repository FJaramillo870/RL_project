#Imported necessary libraries for setting up a gym
import gymnasium as gym

#Created a function named run to keep the game window running
def run():
    #Created the enviornment and specifying the game we want and the render mode.
    #The game is Atari's version of blackjack and human render makes its a visible play window.
    env  = gym.make('ALE/Blackjack-v5', render_mode="human", full_action_space=True)
    #Specifying the frames per second we want
    env.metadata['render_fps']=120
    #Ran this line because it is required before calling step
    state = env.reset()[0]
    #Setting terminated to False for the while loop that is going to be used
    terminated = False

    rewards = 0
    #Setting parameters for the game to run. Stops when it is terminated or when 
    #rewards is larger than -1000
    while(not terminated and rewards>-1000):
        #Created the action_space which is running with sample 
        #for now until controls for the environment are fully working.
        action = env.action_space.sample()
        #Updates an environment with actions returning the 
        #next agent observation, the reward for taking that actions, 
        #if the environment has terminated or truncated due to the 
        #latest action and information from the environment.
        new_state,reward,terminated,_,_ = env.step(action)
        state = new_state
        rewards+=reward
    env.close()

if __name__=='__main__':
    run()