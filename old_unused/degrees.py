#!/usr/bin/env python3

import rg_gs1_10, rg_gs11_20, rg_gs21_30, rg_gs31_40, rg_gs41_50, rg_gs51_60, rg_gs61_70, rg_gs71_80, rg_gs81_90, rg_gs91_100 # Random generated graphs
import cg_gs1_10, cg_gs11_20, cg_gs21_30, cg_gs31_40, cg_gs41_50, cg_gs51_60, cg_gs61_70, cg_gs71_80, cg_gs81_90, cg_gs91_100 # Claude generated graphs
import hc_gs1_10, hc_gs11_20, hc_gs21_30, hc_gs31_40, hc_gs41_50, hc_gs51_60, hc_gs61_70, hc_gs71_80, hc_gs81_90, hc_gs91_100 # Hand-crafted graphs
import test_graph
RG_GRAPH_FILES = [rg_gs1_10, rg_gs11_20, rg_gs21_30, rg_gs31_40, rg_gs41_50, rg_gs51_60, rg_gs61_70, rg_gs71_80, rg_gs81_90, rg_gs91_100]
CG_GRAPH_FILES = [cg_gs1_10, cg_gs11_20, cg_gs21_30, cg_gs31_40, cg_gs41_50, cg_gs51_60, cg_gs61_70, cg_gs71_80, cg_gs81_90, cg_gs91_100]
HC_GRAPH_FILES = [hc_gs1_10, hc_gs11_20, hc_gs21_30, hc_gs31_40, hc_gs41_50, hc_gs51_60, hc_gs61_70, hc_gs71_80, hc_gs81_90, hc_gs91_100]
GRAPH_FILES = [test_graph]

degree_dict = dict()
small_grah_degree_dict = dict()
large_graph_degree_dict = dict()

for f in GRAPH_FILES:
   for g_i in range(len(f.graphs)):
       g = f.graphs[g_i]
       edge_index = g.edge_index
       num_nodes = len(g.x)
       for vert in range(num_nodes):
           degree = int(sum(edge_index[0] == vert)) 
           if degree > 50:
               print("High degree vertex")
               print(f)
               print(g_i)
               print(vert)
           if degree in degree_dict:
               degree_dict[degree] = degree_dict[degree] + 1
           else:
               degree_dict[degree] = 1

           if num_nodes <= 50 and degree in small_grah_degree_dict:
               small_grah_degree_dict[degree] = small_grah_degree_dict[degree] + 1
           elif num_nodes <= 50:
               small_grah_degree_dict[degree] = 1

           if num_nodes > 50 and degree in large_graph_degree_dict:
               large_graph_degree_dict[degree] = large_graph_degree_dict[degree] + 1
           elif num_nodes > 50:
               large_graph_degree_dict[degree] = 1
               
        

tupels = list(degree_dict.items())
tupels.sort()

small_tupels = list(small_grah_degree_dict.items())
small_tupels.sort()

large_tupels = list(large_graph_degree_dict.items())
large_tupels.sort()

print("All degrees")
print(tupels)

print("Small graphs degrees")
print(small_tupels)

print("Large graph degrees")
print(large_tupels)
