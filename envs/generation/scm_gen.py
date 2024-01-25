"""Defines the class for SCM and a generator for generating SCMs"""

from typing import List, Tuple, Any, Callable
import networkx as nx
import dill
from envs.generation.graph_gen import CausalGraphGenerator
import random
from envs.generation.functions import f_linear,  f_interaction,  f_linear_noise


class StructuralCausalModel:
    """The data-generating process behind the scm environments"""
    def __init__(self):
        self.endogenous_vars = {}
        self.exogenous_vars = {}
        self.functions = {}
        self.exogenous_distributions = {}
        self.saved_functions = []

    def add_endogenous_var(self, name: str, value: Any, function: Callable, param_varnames: dict):
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.endogenous_vars[name] = value
        self.functions[name] = (function, param_varnames)

    def add_endogenous_vars(self, vars: List[Tuple[str, Any, Callable, dict]]):
        for v in vars:
            self.add_endogenous_var(v[0], v[1], v[2], v[3])

    def add_exogenous_var(self, name: str, value: Any, distribution: Callable, distribution_kwargs: dict):
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.exogenous_vars[name] = value
        self.exogenous_distributions[name] = (distribution, distribution_kwargs)

    def add_exogenous_vars(self, vars: List[Tuple[str, Any, Callable, dict]]):
        for v in vars:
            self.add_exogenous_var(v[0], v[1], v[2], v[3])

    def add_cause(self, var: str, cause: str, function: Callable):
        assert cause not in self.functions[var][1].keys(), "The the cause you want to add is already a cause."

        # save current causes and add new one
        new_causes = list(self.functions[var][1].keys())
        new_causes.append(cause)

        # delete old var
        self.remove_var(var)

        # add updated var
        self.add_endogenous_var(var, 0.0, function, {c: c for c in new_causes})

    def remove_var(self, name: str):
        if name in self.endogenous_vars.keys():
            assert name in self.endogenous_vars, 'Variable not in list of endogenous vars'

            del self.endogenous_vars[name]
            del self.functions[name]

        else:
            assert name in self.exogenous_vars, 'Variable not in list of exogenous vars'

            del self.exogenous_vars[name]
            del self.exogenous_distributions[name]

    def get_next_instantiation(self) -> Tuple[List, List]:
        """
        Returns a new instantiation of variables consistent with the causal structure and for a sample from the
        exogenous distribution
        :return: Instantiation of endogenous and exogenous variables
        """
        random.seed()
        # update exogenous vars
        for key in self.exogenous_vars:
            if type(self.exogenous_vars[key]) == bool:
                self.exogenous_vars[key] = random.choice([True, False])
            else:
                # TODO: understand why parametrized version below produces always the same sequence for bool
                dist = self.exogenous_distributions[key]
                res = dist[0](**dist[1])
                self.exogenous_vars[key] = res

        # update endogenous vars
        structure_model = self.create_graph()
        node_order = [n for n in nx.topological_sort(structure_model)]
        # propagate causal effects along the topological ordering
        for node in node_order:
            # get the values for the parameters needed in the functions
            params = {}
            for param in self.functions[node][1]:  # parameters of functions
                if self.functions[node][1][param] in self.endogenous_vars.keys():
                    params[param] = self.endogenous_vars[self.functions[node][1][param]]
                else:
                    params[param] = self.exogenous_vars[self.functions[node][1][param]]

            # Update variable according to its function and parameters
            self.endogenous_vars[node] = self.functions[node][0](**params)
        return list(self.endogenous_vars.values()), list(self.exogenous_vars.values())

    def do_interventions(self, interventions: List[Tuple[str, Callable]]):
        """
        Replaces the functions of the scm with the given interventions. Note: only hard interventions supported

        :param interventions: List of tuples
        """
        random.seed()
        self.saved_functions = {}
        for interv in interventions:
            self.saved_functions[interv[0]] = self.functions[interv[0]]
            self.endogenous_vars[interv[0]] = interv[1]()  # this is probably redundant with the next line
            self.functions[interv[0]] = (interv[1], {})

    def undo_interventions(self):
        """
        Restores all functional relations that were deleted in the previous call of do_interventions.
        """
        for key, value in self.saved_functions.items():
            self.functions[key] = value

    def create_graph(self) -> nx.DiGraph:
        """
        Returns the DAG that corresponds to the functional structure of this SCM
        """
        graph = nx.DiGraph()

        # create nodes
        [graph.add_node(var.upper()) for var in self.endogenous_vars]

        for var in self.functions:
            for parent in self.functions[var][1]:
                if parent.lower() in self.endogenous_vars or parent.upper() in self.endogenous_vars:
                    graph.add_edge(parent.upper(), var.upper())

        return graph


class SCMGenerator:
    """Class for generating scms"""
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.all_functions = {
            "linear": f_linear,
            "linear_noise": f_linear_noise,
            "interaction": f_interaction}

    def create_random(self, possible_functions: List[str], n_endo: int, n_exo: int, allow_exo_confounders: bool = False)\
            -> Tuple[StructuralCausalModel, set]:
        """
        Creates and returns a random StructualCausalModel by first creating a fully connected graph and then
        randomly deleting one edge after the other until it is acyclic.

        :return: the random scm and the set of edges that have been removed
        """
        random.seed(self.seed)
        graph_generator = CausalGraphGenerator(n_endo, n_exo, allow_exo_confounders, seed=self.seed)
        graph, removed_edges = graph_generator.generate_random_graph()
        self.seed += 1
        return self.create_scm_from_graph(graph, possible_functions), removed_edges

    def create_scm_from_graph(self, graph: nx.DiGraph, possible_functions: List[str]) -> StructuralCausalModel:
        """Defines how an scm is created from a given causal graph"""
        scm = StructuralCausalModel()
        random.seed(self.seed)

        for n in graph.nodes:
            parents = [p for p in graph.predecessors(n)]
            if graph.nodes[n]['type'] == 'endo':
                # randomly choose one of the possible functions for the current node
                current_function = random.choice(possible_functions)
                scm.add_endogenous_var(n, 0.0, self.all_functions[current_function](parents), {p: p for p in parents})

            else:
                # TODO:  This should be parametrizable
                scm.add_exogenous_var(n, 0.0, random.gauss, {'mu': 0.0, 'sigma': 0.1})

        self.seed += 1
        return scm

    @staticmethod
    def load_dataset(path: str) -> List[StructuralCausalModel]:
        with open(path, 'rb') as f:
            dic = dill.load(f)
        return dic

    # Some pre-defined scms
    # _________________________________________________________________________________________________________________
    @staticmethod
    def make_obs_equ_gauss_3var_envs():
        """
        two 3-var networks with observationally identical distributions for proof of concept
        """
        scm1 = StructuralCausalModel()
        scm1.add_endogenous_vars(
            [('X0', 0.0, lambda: random.gauss(0.0, 0.1), {}),
             ('X1', 0.0, lambda x0: x0, {'x0': 'X0'}),
             ('X2', 0.0, lambda x1: x1, {'x1': 'X1'})])
        scm2 = StructuralCausalModel()
        scm2.add_endogenous_vars(
            [('X0', 0.0, lambda: random.gauss(0.0, 0.1), {}),
             ('X1', 0.0, lambda x0: x0, {'x0': 'X0'}),
             ('X2', 0.0, lambda x0: x0, {'x0': 'X0'})])
        return scm1, scm2

    @staticmethod
    def make_2obj_visual_env():
        """
        Describes an environment with two objects with y and x positions.
        """
        scm = StructuralCausalModel()
        # position of object1 that follows some random noise
        scm.add_endogenous_var(name='X1', value=0.0, function=lambda: 40+random.gauss(0.0, 4.0), param_varnames={})
        scm.add_endogenous_var(name='Y1', value=0.0, function=lambda: 40+random.gauss(0.0, 4.0), param_varnames={})

        # position of object2 that depends on the position of object 1 and some random noise
        scm.add_endogenous_var(name='X2', value=-0.0, function=lambda x1, y1: x1 + y1 + random.gauss(0.0, 4.0),
                               param_varnames={'x1': 'X1', 'y1': 'Y1'})
        scm.add_endogenous_var(name='Y2', value=-0.0, function=lambda x1, y1: x1 + y1 + random.gauss(0.0, 4.0),
                               param_varnames={'x1': 'X1', 'y1': 'Y1'})

        return scm
