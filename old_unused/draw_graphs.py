#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt

import rg_gs1_10, rg_gs11_20, rg_gs21_30, rg_gs31_40, rg_gs41_50, rg_gs51_60, rg_gs61_70, rg_gs71_80, rg_gs81_90, rg_gs91_100 # Random generated graphs
import cg_gs1_10, cg_gs11_20, cg_gs21_30, cg_gs31_40, cg_gs41_50, cg_gs51_60, cg_gs61_70, cg_gs71_80, cg_gs81_90, cg_gs91_100 # Claude generated graphs
import hc_gs1_10, hc_gs11_20, hc_gs21_30, hc_gs31_40, hc_gs41_50, hc_gs51_60, hc_gs61_70, hc_gs71_80, hc_gs81_90, hc_gs91_100 # Hand-crafted graphs
import test_graph
RG_GRAPH_FILES = [rg_gs1_10, rg_gs11_20, rg_gs21_30, rg_gs31_40, rg_gs41_50, rg_gs51_60, rg_gs61_70, rg_gs71_80, rg_gs81_90, rg_gs91_100]
CG_GRAPH_FILES = [cg_gs1_10, cg_gs11_20, cg_gs21_30, cg_gs31_40, cg_gs41_50, cg_gs51_60, cg_gs61_70, cg_gs71_80, cg_gs81_90, cg_gs91_100]
HC_GRAPH_FILES = [hc_gs1_10, hc_gs11_20, hc_gs21_30, hc_gs31_40, hc_gs41_50, hc_gs51_60, hc_gs61_70, hc_gs71_80, hc_gs81_90, hc_gs91_100]
Test_GRAPH_FILES = [test_graph]
GRAPH_FILES = [test_graph]

#p0=red p1=green p2=blue
colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "grey"]

def draw_and_show(graph, show=True, file_name="graph.png"):
    edge_index = graph.edge_index
    edges = []
    for (i,j) in zip(edge_index[0], edge_index[1]):
        edges = edges + [(int(i), int(j))]
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(graph.x))))
    G.add_edges_from(edges)

    graph_colors = []
    for i in graph.x:
        k = i[0]+2*i[1]+4*i[2]
        graph_colors = graph_colors + [colors[int(k)]]
        
    pos = None
    if nx.is_planar(G):
        pos = nx.planar_layout(G)
    #elif nx.is_connected(G):
    #    pos = nx.bfs_layout(G, start = 0)
    else: 
        pos = nx.spectral_layout(G)
    #pos = nx.shell_layout(G) #, pos=pos)
        
    nx.draw(G, node_color=graph_colors, pos=pos)
    plt.savefig("pics_arf/"+file_name)
    if show:
        plt.show()
    plt.clf()


#for i in range(10):
#    for k in range(10):
#        draw_and_show(HC_GRAPH_FILES[i].graphs[k], show=False, file_name="hc_"+str(i)+"_"+str(k))

if __name__ == "__main__":
    while True:
        print("Choose next graph:\n0-RG 1-CG 2-HC 3-Test\n 0-9 graph size\n 0-9 index")
        choice = input()
        choice = choice.split()
        if not len(choice) == 3:
            print("Try again because length is not 3. Length is: "+str(len(choice)))
            print(choice)
            continue
        else:
            rcg = int(choice[0])
            size = int(choice[1])
            i = int(choice[2])
            if rcg < 0 or rcg > 4 or size < 0 or size > 9 or i < 0 or i > 9:
                print("Try again because one of the numbers is out of range")
                continue

            if rcg == 0:
                draw_and_show(RG_GRAPH_FILES[size].graphs[i])
            elif rcg == 1:
                draw_and_show(CG_GRAPH_FILES[size].graphs[i])
            elif rcg == 2:
                draw_and_show(HC_GRAPH_FILES[size].graphs[i])
            else:
                draw_and_show(Test_GRAPH_FILES[size].graphs[i])

