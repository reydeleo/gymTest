import gym.spaces
import time
from Agent import Agent
from Agent import rangefloat
from tqdm import tqdm, trange
import numpy as np
import skimage as skimage
from skimage import transform, color, exposure
from itertools import count
from RLAgents import DQNAgent

import matplotlib.pyplot as plt

ACTIONS = 6 # number of valid actions
OBSERVATION = 32 # timesteps to observe before training
REPLAY_MEMORY = 5000 # number of previous transitions to remember
EXPLORE = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
FRAME_PER_ACTION = 3

NUM_OF_EPISODES = 800

def gameInit(game_state, agent):
    game_state.reset()
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    action = 0                                              #not very sure of this because 0 and also 1 dont do anything
    x_t, r_0, terminal, _ = game_state.step(action)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    x_t = x_t / 255.0


    agent.initialize(x_t.astype(np.float16), do_nothing.astype(np.uint8))



def runEpisode(game_state, agent, training=False):
    gameInit(game_state, agent)
    steps = count()
    episode_reward = 0
    for t in steps:

        if training is True:
            a_t = agent.choose()
        else:
            a_t = agent.choose(0.05)

        # run the selected action and observed next state and reward
        action = np.argmax(a_t)
        x_t1_colored, r_t, terminal,_ = game_state.step(action)
        episode_reward += r_t

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1 / 255.0

        agent.feedback(x_t1.astype(np.float16), r_t, terminal)


        loss = agent.train()

        if terminal is True:
            break

    number_of_steps = next(steps) - 1

    return episode_reward, number_of_steps


def fillMemory(game_state, agent, frames=2**10):
    gameInit(game_state, agent)
    frames = trange(frames)
    frames.set_description('Filling Memory...')
    for t in frames:
        action = np.random.randint(6)                                              #this was changed because now is Pong
        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal, _ = game_state.step(action)
        game_state.render() #this is just for me to see what is happening in the actual game
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1 / 255.0
        agent.feedback(x_t1.astype(np.float16), r_t, terminal)

        if terminal is True:
            gameInit(game_state,agent)


def trainAgentEpisodic(agent):
    # open up a game state to communicate with emulator
    game_state = gym.make('Pong-v0')
    fillMemory(game_state, agent, OBSERVATION)

    episode_scores = []
    with trange(0, NUM_OF_EPISODES) as episodes:
        episodes.set_description('Training...')
        steps = 0
        for episode in episodes:
            _, stp = runEpisode(game_state, agent, training=True)
            steps += stp
            if (episode+1)%20 == 0:
                episodes.set_description('Testing...')
                score, _ = runEpisode(game_state, agent, training=False)
                episode_scores.append(score)
                episodes.set_description('Reward {:.2f} | Epsilon: {:.6f} | Steps {!s} | Training...'.format(np.mean(episode_scores), agent.get_epsilon(), steps))

    print('\nMean: {:.3f} Std: {:.3}'.format(np.mean(episode_scores), np.std(episode_scores)))
    plt.plot(range(0, NUM_OF_EPISODES, 20), episode_scores, 'ro')
    plt.ylabel('Score')
    plt.show()
    print("Episode finished!")
    print("************************")



def main():
    agent = DQNAgent('model.h5', memory_size=REPLAY_MEMORY,
                  epsilon=rangefloat(INITIAL_EPSILON, FINAL_EPSILON, EXPLORE))
    trainAgentEpisodic(agent)




if __name__ == "__main__":
    main()