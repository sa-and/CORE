from abc import ABC, abstractmethod
from typing import Tuple, Union
from agents import CausalAgent
import networkx as nx
import numpy as np


def directed_shd(predicted: Union[np.ndarray, nx.DiGraph], target: nx.DiGraph) -> int:
    if type(predicted) == np.ndarray:
        predicted = nx.from_numpy_array(predicted, create_using=nx.DiGraph)
        nx.relabel_nodes(predicted, mapping={i: "X"+str(i) for i in range(len(predicted.nodes()))}, copy=False)
    else:
        assert len(predicted.nodes) == len(target.nodes), 'Graphs need to have the same amount of nodes'

    # this corresponds to the SHD (structural Hamming distance or SHD (Tsamardinos et al., 2006;
    # with the difference that we consider undirected edges as bidirected edges and flip is considered
    # as 2 errors instead of 1
    differences = 0
    for node in predicted.adj:
        # check which edges are too much in the predicted graph
        for parent in predicted.adj[node]:
            if not parent.upper() in target.adj[node.upper()]:
                differences += 1
        # check which edges are missing in the predicted graph
        for parent in target.adj[node.upper()]:
            if not parent.lower() in predicted.adj[node]:
                differences += 1

    return differences


class EvalFunc(ABC):
    """Interface for evaluation function for the RL learning"""
    effect_threshold: float
    steps_this_episode: int

    def __init__(self, agent: CausalAgent):
        super(EvalFunc, self).__init__()
        self.agent = agent
        self.steps_this_episode = 0

    @abstractmethod
    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Calculates for each step whether the episode is done and what the reward for that step is

        :param action_successful: whether the action which was taken was executed successfully
        :param allow_unsuccessful_actions: Whether unsuccessful actions are allowed
        :return: Whether the episode is done and the reward of the current step
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_model(self):
        """Defines how the model is evaluated"""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Functions evaluating the model based on the structural difference to the target graph
class StructureEvalFunc(EvalFunc):
    def __init__(self, agent: CausalAgent, graph: nx.DiGraph):
        super(StructureEvalFunc, self).__init__(agent)
        self.compare_graph = graph

    def set_compare_graph(self, graph: nx.DiGraph):
        self.compare_graph = graph

    def _eval_model(self):
        diff = directed_shd(self.agent.causal_model, self.compare_graph)
        return -diff

    @abstractmethod
    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Calculates for each step whether the episode is done and what the reward for that step is

        :param action_successful: whether the action which was taken was executed successfully
        :param allow_unsuccessful_actions: Whether unsuccessful actions are allowed
        :return: Whether the episode is done and the reward of the current step
        """
        raise NotImplementedError


class FixedLengthStructEpisode(StructureEvalFunc):
    def __init__(self, agent: CausalAgent, graph: nx.DiGraph, ep_length: int):
        super(FixedLengthStructEpisode, self).__init__(agent, graph)
        self.episode_length = ep_length

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        self.steps_this_episode += 1

        done = False
        # Evaluate when the episode length is reached
        if self.steps_this_episode >= self.episode_length-1:
            done = True
            self.steps_this_episode = 0
        reward = self._eval_model()

        if not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -100

        return done, reward


class NoEval(EvalFunc):
    """Does nothing. Used when applying the policy so there are no prints and rewards"""

    def _eval_model(self):
        pass

    def __init__(self):
        super(NoEval, self).__init__(None)

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        return False, 0.0
