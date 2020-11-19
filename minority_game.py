# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:04:54 2020

@author: eelil
"""

import numpy as np

def lastMBits(n, m):
    return n & ((1 << m) - 1)

def insertRightBit(n, bit):
    return (n << 1) | bit

# N is the number of players, m is the length of memory
def runMinorityGame(N, m):
    m = int(m)
    n_strategies = 2
    
    past = 0 # The past actions are stored in the bits, right-most bit is the latest action
    
    strategies = np.random.randint(low=0, high=2, size=(N, n_strategies, 2**m))
    performance = np.zeros((N, n_strategies))
    chosenStrategy = np.zeros(N)
    
    n_iterations = 1000
    n1_values = np.zeros(n_iterations)
    
    for k in range(n_iterations):
        # If the performance of the strategies are equal, the first one is chosen
        chosenStrategy = np.argmax(performance, axis=1) # Shape (N,)
        actions = strategies[np.arange(N), chosenStrategy, :] # Shape (N,2**m)
        past_index = lastMBits(past, m)
        chosen_actions = actions[:, past_index] # Shape (N,)
        
        n1 = sum(chosen_actions)
        n1_values[k] = n1
        action = 1 if n1 < N/2 else 0
        past = insertRightBit(past, action)
        performance += strategies[:, :, past_index] == action
        
        # print(past_index)
        # print(chosen_actions)
        # print(action)
        # print(past)
        # print(strategies)
        # print(performance)
        
    sigma = np.std(n1_values)
    return sigma

if __name__ == '__main__':

    save = False    

    n_games = 10
    N_vals = np.array([51, 71, 101, 151, 201, 251])
    m_vals = np.arange(2, 13)
    sigmas = np.zeros((len(N_vals), len(m_vals), n_games))
    
    for j, N in enumerate(N_vals):
        print('\nN = {}\n'.format(N))
        for i, m in enumerate(m_vals):
            print(i, '/', len(m_vals))
            for n in range(n_games):
                sigmas[j, i, n] = runMinorityGame(N, m)
            
    if save:
        np.savez('minority_game_results.npz', sigmas=sigmas, N_vals=N_vals, m_vals=m_vals)