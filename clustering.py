# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:15:22 2020

@author: eelil
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

N = 101

# l = linkage(strategies, metric='hamming', optimal_ordering=True)
# print(l)

for threshold in np.linspace(0.1, 0.5, 5):
    
    gcc_sizes = []

    for m in range(2,12):
        
        strategies = np.random.randint(low=0, high=2, size=(N, 2**m))
    
        g = nx.Graph()
        
        for i, p1 in enumerate(strategies):
            for j, p2 in enumerate(strategies):
                hamming = sum(abs(p1-p2))/len(p1)
                # g.add_edge(i, j, weight=hamming)
                
                if hamming <= threshold:
                    g.add_edge(i,j)
                
        # mst = nx.minimum_spanning_tree(g)
        # pos = nx.spring_layout(mst)
        # plt.figure()
        # nx.draw(mst, pos)
                    
        gcc = sorted(nx.connected_components(g), key=len, reverse=True)
        gcc_sizes.append(len(gcc[0]))
        
    plt.plot(range(2,12), np.array(gcc_sizes)/N, 'x', label=threshold)

plt.legend()