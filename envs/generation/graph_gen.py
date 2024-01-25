"""Classes for generating causal DAGs"""

from typing import Tuple, List
from scipy.special import comb
import copy
import random
import networkx as nx
from tqdm import tqdm
import pickle


class CausalGraphGenerator:
    def __init__(self, n_endo: int, n_exo: int = 0, allow_exo_confounders: bool = False, seed: int = 42):
        self.allow_exo_confounders = allow_exo_confounders
        self.seed = seed

        # create var names
        self.endo_vars = ['X' + str(i) for i in range(n_endo)]
        self.exo_vars = ['U' + str(i) for i in range(n_exo)]

        # determine potential causes for each endogenous var
        self.potential_causes = {}
        exo_copy = copy.deepcopy(self.exo_vars)
        for v in self.endo_vars:
            # if confounding is allowed, any node can be the parent of an endogenous var
            if allow_exo_confounders:
                self.potential_causes[v] = self.endo_vars + self.exo_vars
            # else, only one exo var can be the cause of an endo var
            else:
                if not len(exo_copy) == 0:
                    self.potential_causes[v] = self.endo_vars + [exo_copy.pop()]
                else:
                    self.potential_causes[v] = self.endo_vars + []
            self.potential_causes[v].remove(v)

        del exo_copy
        self.fully_connected_graph = self._make_fully_connected_dag()

    def _make_fully_connected_dag(self) -> nx.DiGraph:
        """
        Creates and returns a fully connected graph. In this graph the exogenous variables have only outgoing edges.
        """
        graph = nx.DiGraph()
        [graph.add_node(u, type='exo') for u in self.exo_vars]
        [graph.add_node(v, type='endo') for v in self.endo_vars]
        for n, causes in self.potential_causes.items():
            [graph.add_edge(c, n) for c in causes]
        return graph

    def generate_random_graph(self, method: str = 'ER', p: float = 0.1) -> Tuple[nx.DiGraph, set]:
        """
        Creates and returns a random nx.DiGraph either by first creating a fully connected graph and then
        randomly deleting one edge after the other until it is acyclic, or creating an Edos Reny graph.

        :return: the random nx.DiGRaph and the set of edges that have been removed from a fully connected graph
        """
        if method == 'ER':  #create erdos reny graph
            graph, removed_edges = self._create_er_graph(p)
        else:
            graph, removed_edges = self._create_from_fully_connected()

        # create fcm
        return graph, removed_edges

    def _create_er_graph(self, p: float = 0.1) -> (nx.DiGraph, List[Tuple[str, str]]):
        graph = nx.fast_gnp_random_graph(len(self.endo_vars), p=p, directed=True)

        # rename the nodes and types according to the fully connected graph
        graph = nx.relabel_nodes(graph, {n: m for (n, m) in zip(graph.nodes, self.fully_connected_graph.nodes)})
        nx.set_node_attributes(graph,
                               {n: {'type': self.fully_connected_graph.nodes()[m]['type']} for (n, m) in zip(graph.nodes, self.fully_connected_graph.nodes)})

        # make acyclic
        while not nx.is_directed_acyclic_graph(graph):
            random_edge = random.choice(list(graph.edges()))
            graph.remove_edge(random_edge[0], random_edge[1])

        # compute which edges are removed
        removed_edges = nx.difference(self.fully_connected_graph, graph).edges

        return graph, set(removed_edges)

    def _create_from_fully_connected(self) -> (nx.DiGraph, List[Tuple[str, str]]):
        # generate fully connected graph
        graph = copy.deepcopy(self.fully_connected_graph)
        # delete random edges from the endogenous subgraph until acyclic
        removed_edges = set()
        random.seed(self.seed)
        while not nx.is_directed_acyclic_graph(graph):
            random_edge = random.sample(graph.edges, 1)[0]
            self.seed += 1
            removed_edges.add(random_edge)
            graph.remove_edge(random_edge[0], random_edge[1])
        return graph, removed_edges


class CausalGraphSetGenerator:
    """
    Represents a collection of unique DAGs and facilitates their bookkeeping
    """
    def __init__(self, n_endo: int, n_exo: int = 0, allow_exo_confounders: bool = False, seed: int = 42):
        self.generator = CausalGraphGenerator(n_endo, n_exo=n_exo, allow_exo_confounders=allow_exo_confounders, seed=seed)
        self.graphs = []
        self.max_endo_dags = CausalGraphSetGenerator.max_n_dags(n_endo)

    def generate(self, n: int, method: str = 'ER', p: float = 0.1):
        """
        Generate n distinct causal DAGs
        :param n: how many DAGs to create
        :param method: which method to use to generate the graphs.
        :param p: probability with which an edge is added if ER is used.
        """
        # check whether more DAGs are to be created then combinatorically possible. Only do this if n is not
        # too big because computation takes forever for n > 20 and for such values there exist over 2.3e+72
        # different graphs
        n_exo = len(self.generator.exo_vars)
        if n > self.max_endo_dags * (n_exo+1):
            n = self.max_endo_dags * (n_exo+1)
            print('Only ',  self.max_endo_dags * (n_exo+1), ' graphs can be created.')

        self.graphs = []
        rem_edges_list = []
        resampled = 0
        print('Creating graphs...')
        pbar = tqdm(total=n - 1)
        while len(self.graphs) < n - 1:
            graph, rem_edges = self.generator.generate_random_graph(method=method, p=p)
            if any([rem_edges == other for other in rem_edges_list]):
                resampled += 1
                continue
            else:
                self.graphs.append(graph)
                rem_edges_list.append(rem_edges)
                pbar.update(1)
        pbar.close()
        print(resampled, 'models resampled')

    def save(self, filepath: str, mode: str = 'wb'):
        with open(str(filepath), str(mode)) as f:
            pickle.dump(self.graphs, f)

    def load(self, filepath: str):
        # Warning: doesn't check whether the loaded graphs are actually a set
        with open(str(filepath), 'rb') as f:
            self.graphs = pickle.load(f)

    @staticmethod
    def max_n_dags(n_vertices: int) -> int:
        """
        Computes the maximal number of different DAGs over n_vertices nodes. Implemented as in Robinson (1973)

        :param n_vertices:
        :return: max number of dags
        """
        if n_vertices < 0:
            return 0
        elif n_vertices == 0:
            return 1
        else:
            summ = 0
            for k in range(1, n_vertices + 1):
                summ += (-1) ** (k - 1) * comb(n_vertices, k) * 2 ** (
                        k * (n_vertices - k)) * CausalGraphSetGenerator.max_n_dags(n_vertices - k)
            return int(summ)
