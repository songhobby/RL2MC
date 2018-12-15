#!/usr/bin/python
from z3 import *
from mypdr0 import PDR
import time
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

STEPS = 7


class QL:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# SAFE
# Similar to OneAtATime
# A boolean bit vector is initialized with size 8
# to TTTTTTTT. One bit can be flipped per frame but 
# now the two neighbors to it's left must also flip for a total of three.
# The post condition is that at least one bool is True
# which cannot be violated with a bit vector of size 8 and three bits flipped per frame
def ThreeAtATimeEven():
    size = 8
    variables = [Bool(str(i)) for i in range(size)]
    primes = [Bool(str(i) + '\'') for i in variables]

    def triple(i):
        return And(*[primes[j] == variables[j] for j in range(size) if (j != i and j != i-1 and j != i-2)]+\
            [Not(primes[i] == variables[i]),Not(primes[i-1] == variables[i-1]),Not(primes[i-2] == variables[i-2])])

    init = And(*[variables[i] for i in range(size-1)] + [(variables[-1])])
    trans = Or(*[triple(i) for i in range(size)])
    post = Or(*variables)

    return (variables, primes, init, trans, post)

# UNSAFE
# Initialize a boolean bitfield [AAAAA BBBBB]
# Each iteration, add the value of BBBBB to AAAAA
# incrementing it
# In this example, BBBBB is 00001 and the postcondition is that
# AAAAA is not 11111, which is unsafe after 16 frames
def BooleanIncrementer():
    len = 8
    variables = [Bool(str(i)) for i in range(len)]
    primes = [Bool(str(i) + '\'') for i in variables]
    init = And(*[Not(variables[i]) for i in range(len-1)] + [variables[-1]])
    def carryout(pos):
        if pos==len/2:
            return False
        else:
            return Or(And(Xor(variables[pos],variables[pos+len/2]), carryout(pos+1)),And(variables[pos],variables[pos+len/2]))
    trans = And(*[primes[i] == Xor(Xor(variables[i],variables[i+len/2]),carryout(i+1)) for i in range(len/2)] \
        + [primes[i+len/2] == variables[i+len/2] for i in range(len/2)])
    post = Not(And(*[variables[i] for i in range(len/2)]))
    return (variables, primes, init, trans, post)

# SAFE
# Add overflow protection to the previous boolean incrementer
# When the incrementer becomes full, it will not add any more to it
# There is an overflow bit that gets set if there is any carryover from the MSB
# so the postcondition is Not(overflow)
def IncrementerOverflow():
    size = 8
    overflow = Bool('Overflow')
    variables = [Bool(str(i)) for i in range(size)] + [overflow]
    primes = [Bool(str(i) + '\'') for i in variables]
    overflowprime = primes[-1]
    init = And(*[Not(variables[i]) for i in range(size-1)] + [variables[size-1], overflow == False])
    def carryout(pos):
        if pos==size/2:
            return False
        else:
            return Or(And(Xor(variables[pos],variables[pos+size/2]), carryout(pos+1)),And(variables[pos],variables[pos+size/2]))
    trans = If(And(*[variables[i] for i in range(size/2)]), \
        And(*[variables[i] == primes[i] for i in range(len(variables))]),
        And(*[primes[i] == Xor(Xor(variables[i],variables[i+size/2]),carryout(i+1)) for i in range(size/2)] \
            + [primes[i+size/2] == variables[i+size/2] for i in range(size/2)] \
            + [overflowprime==carryout(0)])\
        )
    post = Not(overflow)
    return (variables, primes, init, trans, post)


if __name__ == "__main__":
#    state_size = 10
#    action_size = 8
#    agent = QL(state_size, action_size)
#    print("Experiment1")
#    for i in range(2):
#        start = time.time()
#        solver = PDR(*ThreeAtATimeEven())
#        solver.run(agent)
#        print(time.time()-start-solver.ignore)
    state_size=10
    action_size=8
    agent = QL(state_size, action_size)
    print("Experiment2")
    for i in range(100):
        start = time.time()
        solver = PDR(*BooleanIncrementer())
        solver.run(agent)
        print(time.time()-start-solver.ignore)
    state_size=10
    action_size=9
    agent = QL(state_size, action_size)
    print("Experiment3")
    for i in range(100):
        start = time.time()
        solver = PDR(*IncrementerOverflow())
        solver.run(agent)
        print(time.time()-start-solver.ignore)
        
