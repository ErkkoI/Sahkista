# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:04:54 2020

@author: eelil
"""

import numpy as np
import matplotlib.pyplot as plt

def lastMBits(n, m):
    return n & ((1 << m) - 1)

def insertRightBit(n, bit):
    return (n << 1) | bit

# N is the number of players, m is the length of memory
def runMinorityGame(N, m, n_iterations):
    m = int(m)
    n_strategies = 2
    
    past = 0 # The past actions are stored in the bits, right-most bit is the latest action
    
    strategies = np.random.randint(low=0, high=2, size=(N, n_strategies, 2**m))
    performance = np.zeros((N, n_strategies))
    chosenStrategy = np.zeros(N)
    
    n1_values = np.zeros(n_iterations)
    
    histories = np.zeros(2**m)
    p1_given_mu = np.zeros(2**m)
    n1_given_mu = np.zeros(2**m)
    
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
        
        p1_given_mu[past_index] += action        
        n1_given_mu[past_index] += n1
        histories[past_index] += 1
        
        # for mu in range(2**m):
        #     chosen_actions_mu = actions[:, mu]
        #     n1_mu = sum(chosen_actions_mu)
        #     action_mu = 1 if n1_mu < N/2 else 0
        #     p1_given_mu[mu] += action_mu
        
        # print(past_index)
        # print(chosen_actions)
        # print(action)
        # print(past)
        # print(strategies)
        # print(performance)
        
        
    # H = np.nanmean((2*n1_given_mu / histories - N)**2)
    
    
    p1_given_mu /= histories
    H = np.nanmean((p1_given_mu - 0.5)**2)

        
    sigma = np.std(n1_values)
    
    return sigma, H, p1_given_mu, n1_values

if __name__ == '__main__':
    
    time_series = False
    p_values = True
    scaling_results = False
    
    if scaling_results:

        save = True   
    
        n_games = 50
        N_vals = np.array([51, 71, 101, 151, 201, 251])
        m_vals = np.arange(2, 13)
        # n_games = 1
        # N_vals = [251]
        # m_ = 4
        # m_vals = [m_]
        
        sigmas = np.zeros((len(N_vals), len(m_vals), n_games))
        
        H_vals = np.zeros((len(N_vals), len(m_vals), n_games))
        
        n_iterations = 1000
        
        for j, N in enumerate(N_vals):
            print('\nN = {}\n'.format(N))
            for i, m in enumerate(m_vals):
                print(i, '/', len(m_vals))
                for n in range(n_games):
                    sigma, H, _, _ = runMinorityGame(N, m, n_iterations)
                    sigmas[j, i, n] = sigma
                    H_vals[j, i, n] = H
                
        if save:
            np.savez('minority_game_results_5000.npz', sigmas=sigmas,
                      N_vals=N_vals, m_vals=m_vals, H_vals=H_vals)
            
    if time_series:
        
        N = 251
        m_vals = [2, 5, 12]
        n_iterations = 500
        plt.figure()
        
        for i, m in enumerate(m_vals):
            _, _, _, time_series = runMinorityGame(N, m, n_iterations)
            plt.subplot(len(m_vals), 1, i+1)
            plt.plot(time_series, label='m={}'.format(m))
            plt.ylim([75,175])
            plt.grid()
            plt.xlabel('Iteration')
            plt.ylabel('n_1')
            
            plt.legend()
            
    if p_values:
        
        N = 101
        m_vals = [5,6]
        n_iterations = 50000
        plt.figure()
        
        for i, m in enumerate(m_vals):
            _, _, p_vals, _ = runMinorityGame(N, m, n_iterations)
            plt.subplot(len(m_vals), 1, i+1)
            plt.bar(np.arange(2**m), p_vals, label='m={}'.format(m))
            plt.ylim([0,1])
            plt.grid()
            plt.xlabel('$\mu$')
            plt.ylabel('P(1|$\mu$)')
            plt.legend()