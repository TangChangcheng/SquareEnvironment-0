import numpy as np
import gym
import json

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from  Environment import Env, status, actions


# ENV_NAME = 'CartPole-v0'
ENV_NAME = 'SQ-v0'


# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
env = Env()

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
env.is_train = True

dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

env.is_train = False
env.plot_row = 1
env.plot_col = 5

# with open('dqn_action.json', 'w') as fw:
#
#     observation = [i / 100 - 3 for i in range(600)]
#     action = [float(actions[dqn.forward(np.array([obs]))]) for obs in observation]
#     json.dump({'observation': observation, 'action': action}, fw)
#
# with open('dqn_qvalue.json', 'w') as fw:
#     state_batch = status.reshape([-1, 1, 1])
#     q_val = dqn.compute_batch_q_values(state_batch)
#     q = {'status': state_batch.tolist(), 'q_value': q_val.tolist()}
#     json.dump(q, fw)


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

env.plt.ioff()
env.plt.show()
