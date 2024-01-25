"""Script ot generate and save a dataset of DAGs and resulting SCMs"""

import argparse
import os
import random

from envs.generation.graph_gen import CausalGraphSetGenerator
import pickle

from networkx.readwrite import json_graph
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-test-graphs', type=int, default=3, help='Number of test graphs to generate.')
    parser.add_argument('--n-train-graphs', type=int, default=10,  help='Number of training graphs to generate.')
    parser.add_argument('--save-dir', type=str, default='/data/delme/',
                        help='Filepath of where to save the data.')
    parser.add_argument('--n-endo', type=int, default=3,
                        help='Amount of endogenous variables.')
    parser.add_argument('--n-exo', type=int, default=0,
                        help='Amount of exogenous variables.')
    parser.add_argument('--method', type=str, choices=['dense', 'ER'], default='ER',
                        help='pick the method with which the graphs should be generated')
    parser.add_argument('--edge-probability', type=float, default=0.1, help='define the probability of an edge being'
                                                                            'in the graph. Only for ER graphs.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    args = parser.parse_args()

    directory = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(directory + args.save_dir):
        os.makedirs(directory + args.save_dir)
    #os.mkdir(args.save_dir)

    # Generate the graphs
    graph_set_gen = CausalGraphSetGenerator(n_endo=args.n_endo, n_exo=args.n_exo, allow_exo_confounders=False, seed=args.seed)
    print('Generating Graphs...')
    graph_set_gen.generate(args.n_train_graphs+args.n_test_graphs, args.method)

    random.shuffle(graph_set_gen.graphs)

    with open(directory + args.save_dir+'\\all_graphs.pkl', 'wb') as f:
        pickle.dump(graph_set_gen.graphs, f)
    js_graphs = [json_graph.node_link_data(G) for G in graph_set_gen.graphs]
    json_obj = json.dumps(js_graphs)
    with open(directory + args.save_dir+'\\all_graphs_json.JSON', 'w') as f:
        f.write(json_obj)

    with open(directory + args.save_dir+'\\train.pkl', 'wb') as f:
        pickle.dump(graph_set_gen.graphs[:args.n_train_graphs], f)
    js_graphs = [json_graph.node_link_data(G) for G in graph_set_gen.graphs[:args.n_train_graphs]]
    json_obj = json.dumps(js_graphs)
    with open(directory + args.save_dir + '\\train_json.JSON', 'w') as f:
        f.write(json_obj)

    with open(directory + args.save_dir+'\\test.pkl', 'wb') as f:
        pickle.dump(graph_set_gen.graphs[args.n_train_graphs:], f)
    js_graphs = [json_graph.node_link_data(G) for G in graph_set_gen.graphs[args.n_train_graphs:]]
    json_obj = json.dumps(js_graphs)
    with open(directory + args.save_dir + '\\test_json.JSON', 'w') as f:
        f.write(json_obj)

    print('Graphs saved to ', directory + args.save_dir)
