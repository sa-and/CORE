from typing import Tuple, List, NoReturn, Union, Any
from abc import ABC, abstractmethod
from causalnex.structure import StructureModel
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations, permutations
import random
from gym.spaces import Box, MultiDiscrete
import warnings


class CausalAgent(ABC):
    """
    The base class for all agents which maintain an epistemic causal graph about their environment.
    """
    var_names: Union[int, List[str]]
    causal_model: StructureModel
    collected_data: dict
    actions: List[Any]
    state_repeats: int

    def __init__(self, vars: Union[int, List[str]],
                 causal_graph: StructureModel = None,
                 state_repeats: int = 1,
                 allow_interventions: bool = True):
        self.allow_interventions = allow_interventions
        if type(vars) == int:
            self.var_names = ['x' + str(i) for i in range(vars)]
        else:
            self.var_names = vars

        # initialize causal model
        if causal_graph:
            self.causal_model = causal_graph
        else:
            self.causal_model = StructureModel()
            [self.causal_model.add_node(name) for name in self.var_names]
            self.reset_causal_model()

        # initialize the storages for observational and interventional data.
        self.collected_data = {}

        self.action_space = None
        self.observation_space = None
        self.actions = []
        self.current_action = None
        self.state_repeats = state_repeats

    # --------------------------- Methods for maintaining the causal structure of the agent ---------------------------
    def set_causal_model(self, causal_model: StructureModel):
        self.causal_model = causal_model

    def reset_causal_model(self, mode: str = 'empty'):
        """
        Sets the causal graph of the agent to either a graph with random edges or without edges at all.
        :param mode: 'random' or 'empty'
        """
        all_pairs = [(v[0], v[1]) for v in permutations(self.var_names, 2)]

        if mode == 'random':
            random.shuffle(all_pairs)
            for p in all_pairs:
                self.update_model(p, random.choice([0, 1, 2]))

        elif mode == 'empty':
            # delete all edges
            for p in all_pairs:
                self.update_model(p, 0)
        else:
            raise TypeError('No reset defined for mode ' + mode)

    def update_model(self, edge: Tuple[str, str],
                     manipulation: int,
                     allow_disconnecting: bool = True,
                     allow_cycles: bool = True) -> bool:
        """
        Updates model according to action and returns the success of the operation. Reversing and removing an edge that
        doesn't exists has no effect. Adding an edge which already exists has no effect.

        :param edge: The edge to be manipulated. e.g. (X0, X1)
        :param manipulation: 0 = remove edge, 1 = add edge, 2 = reverse edge
        :param allow_disconnecting: If true, manipulations which disconnect the causal graph can be executed.
        :param allow_cycles: If true, manipulations which result in a cycle can be executed.
        :return: True if the manipulation was successful. False if it wasn't or it was illegal according to
        'allow_disconnecting' or 'allow_cycles'.
        """

        if manipulation == -1:  # remove edge if exists
            if self.causal_model.has_edge(edge[0], edge[1]):
                self.causal_model.remove_edge(edge[0], edge[1])
                removed_edge = (edge[0], edge[1])
            else:
                return False

            # disconnected graph
            if not allow_disconnecting and nx.number_weakly_connected_components(self.causal_model) > 1:
                self.causal_model.add_edge(removed_edge[0], removed_edge[1])
                return False

        elif manipulation == 1:  # add edge
            if not self.causal_model.has_edge(edge[0], edge[1]):  # only add edge if not already there
                self.causal_model.add_edge(edge[0], edge[1])
            else:
                return False

            if not nx.is_directed_acyclic_graph(self.causal_model) and not allow_cycles:  # check if became cyclic
                self.causal_model.remove_edge(edge[0], edge[1])
                return False

        return True

    def display_causal_model(self) -> NoReturn:
        fig, ax = plt.subplots()
        nx.draw_circular(self.causal_model, ax=ax, with_labels=True)
        fig.show()

    def get_graph_state(self) -> List[float]:
        """
        Return the flattened adj matrix of the agents causal model

        :return: state of the graph
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nx.adjacency_matrix(self.causal_model).todense().flatten().tolist()

    # ---------------------------------------------- Abstract methods ----------------------------------------------
    @abstractmethod
    def get_action_from_actionspace_sample(self, sample: Any):
        raise NotImplementedError

    @abstractmethod
    def store_observation_per_action(self, obs: List[Any]):
        raise NotImplementedError

    @abstractmethod
    def update_model_per_action(self, action: Any):
        raise NotImplementedError


class DiscreteAgentInterventionStructure(CausalAgent):
    """
    Agent that performs interventions and structure actions at the same time.
    """
    current_mode: str
    action_space: MultiDiscrete

    def __init__(self, n_vars: int,
                 causal_graph: StructureModel = None,
                 state_repeats: int = 20,
                 interv_value: Union[int, float] = 20,
                 allow_interventions: bool = True):
        super(DiscreteAgentInterventionStructure, self).__init__(n_vars, causal_graph, state_repeats=state_repeats,
                                                                 allow_interventions=allow_interventions)
        # create a list of interventions
        if self.allow_interventions:
            self.int_actions = [(i, interv_value) for i in range(n_vars)]

        # for updating the adj. matrix
        # where operation can be one of: delete = -1, add = 1, no action meas keep that edge as it is
        edges = [e for e in combinations(self.var_names, 2)]
        edges.extend([(e[1], e[0]) for e in edges])
        edges.sort()
        self.adj_actions = []
        for i in [-1, 1]:
            self.adj_actions.extend([(edge, i) for edge in edges])
        self.int_actions.append(None)
        self.adj_actions.append(None)
        self.current_action = (None, None)
        #                                   n_vars + 1                  2(n_vars^2 - n_vars)
        self.action_space = MultiDiscrete([len(self.int_actions), len(self.adj_actions)])
        #                                         episode_length * (sample+intervention_target+ none_intervention) + adj + time_left)
        self.observation_space = Box(-50.0, 50.0, ((state_repeats * int(2*n_vars+1) + n_vars**2 + 1),))

    def store_observation_per_action(self, obs: List[Union[bool, float]]):
        """Takes an observation and stores it in the right data frame"""
        if self.current_action[0] == 1 or self.current_action[0] == None:  # no itervention
            self.store_observation(obs, None, None)
        else:
            self.store_observation(obs, self.var_names[self.current_action[1]], self.current_action[2])

    def update_model_per_action(self, action) -> bool:
        """Updates model according to action and returns the success of the operation"""
        edge = action[0]
        manipulation = action[1]

        return self.update_model(edge, manipulation)

    def get_action_from_actionspace_sample(self, sample: Tuple[int, int]):
        return self.int_actions[sample[0]], self.adj_actions[sample[1]]


