# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:32:28 2020

@author: eelil
"""

import numpy as np
import matplotlib.pyplot as plt

save = True

data = np.load('minority_game_results_new.npz')
# time_series = data['time_series']
sigmas = data['sigmas'] # Shape (|N_vals|, |m_vals|, |n_games|)
N_vals = data['N_vals']
m_vals = data['m_vals']
# p_vals = data['p_vals']
H_vals = data['H_vals']

sigma_avgs = np.mean(sigmas, axis=2)
sigma_stds = np.std(sigmas, axis=2)
H_avgs = np.mean(H_vals, axis=2)



def plot_results(x, y, style, xlabel, ylabel, log_axis=False):
    fig = plt.figure()
    
    if len(x.shape) == 1:
        for j, N in enumerate(N_vals):
            plt.plot(x, y[j], style, label=f'N={N}')
            
    elif len(x.shape) == 2:
        for j, N in enumerate(N_vals):
            plt.plot(x[j], y[j], style, label=f'$N={N}$')
            
    if log_axis:
        plt.xscale('log')
        plt.yscale('log')
        
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    return fig

fig1 = plot_results(m_vals, sigma_avgs, style='--', xlabel='$m$', ylabel='$\sigma$')
fig2 = plot_results(np.outer(1/N_vals, 2**m_vals), sigma_avgs**2/N_vals[:, np.newaxis],
                    style='--', xlabel='$2^m / N$', ylabel='$\sigma^2 / N$', log_axis=True)
plt.hlines(0.25, 0, 100)
fig3 = plot_results(m_vals, sigma_stds/N_vals[:, np.newaxis], style='--', xlabel='$m$', ylabel='$\sigma^2 / N$')

fig4 = plot_results(np.outer(1/N_vals, 2**m_vals), np.sqrt(H_avgs),
             style='--', xlabel='$2^m / N$', ylabel='$H$')
plt.xscale('log')

if save:

    fig1.savefig('sigma_v_m.pdf')
    fig2.savefig('scaling.pdf')
    fig3.savefig('sigma_spread.pdf')
    fig4.savefig('predictability.pdf')
