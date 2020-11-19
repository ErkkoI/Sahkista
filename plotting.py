# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:32:28 2020

@author: eelil
"""

import numpy as np
import matplotlib.pyplot as plt

save = False

data = np.load('minority_game_results.npz')
sigmas = data['sigmas'] # Shape (|N_vals|, |m_vals|, |n_games|)
N_vals = data['N_vals']
m_vals = data['m_vals']

sigma_avgs = np.mean(sigmas, axis=2)
sigma_stds = np.std(sigmas, axis=2)

def plot_results(x, y, style, xlabel, ylabel):
    fig = plt.figure()
    
    if len(x.shape) == 1:
        for j, N in enumerate(N_vals):
            plt.plot(x, y[j], style, label='N={}'.format(N))
            
    elif len(x.shape) == 2:
        for j, N in enumerate(N_vals):
            print(y)
            plt.plot(x[j], y[j], style, label='N={}'.format(N))
        
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    return fig

fig1 = plot_results(m_vals, sigma_avgs, style='-', xlabel='m', ylabel='Sigma')
fig2 = plot_results(np.outer(1/N_vals, 2**m_vals), sigma_avgs**2/N_vals[:, np.newaxis], style='--', xlabel='2^m / N', ylabel='Sigma^2 / N')
fig3 = plot_results(m_vals, sigma_stds, style='--', xlabel='m', ylabel='Standard deviation')

if save:

    fig1.savefig('sigma_v_m.pdf')
    fig2.savefig('scaling.pdf')
    fig3.savefig('sigma_spread.pdf')
