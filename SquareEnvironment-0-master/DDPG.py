import numpy as np
import pandas as pd
import gym
import json

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from Environment import Env, status, actions

import matplotlib.pyplot as plt

gym.undo_logger_setup()


ENV_NAME = 'SQ'
env = Env()
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

actor_depth = 4
actor_width = 32

critic_depth = 6
critic_width = 64

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
for k in range(actor_depth):
    actor.add(Dense(actor_width))
    actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
for k in range(critic_depth):
    x = Dense(critic_width)(x)
    x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(outputs=x, inputs=[action_input, observation_input])
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=0.99,
                  target_model_update=1e-3,
                  delta_clip=10)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# agent.load_weights('sq_{}_weights.h5f'.format(ENV_NAME))
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.load_weights('ddpg_fixed_weights_{}_weights.h5f'.format(ENV_NAME))

# env.is_train = True
# agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=20)

# After training is done, we save the final weights.
# agent.save_weights('ddpg_fixed_weights_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.

env.is_train = False
env.plot_row = 1
env.plot_col = 5

q_values = pd.DataFrame()
st = status.reshape([-1, 1])
for action in actions:
    state1_batch_with_action = [st, np.ones(st.shape).reshape(-1, 1, 1) * action]
    q_values = pd.concat([q_values, pd.DataFrame(agent.target_critic.predict_on_batch(state1_batch_with_action))], axis=1)
q_values.to_csv('critic.csv')


with open('actor.json', 'w') as fw:
    observation = status.tolist()
    action = [float(agent.forward(np.array([obs]))[0]) for obs in observation]
    json.dump({'observation': observation, 'action': action}, fw)


agent.test(env, nb_episodes=5, visualize=True, verbose=1, nb_max_episode_steps=20)

env.plt.ioff()
env.plt.show()