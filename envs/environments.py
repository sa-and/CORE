"""Defines the classes for the Gym environments that use an SCM"""
import random
from typing import List, Callable, Tuple, NoReturn, Optional, Union

import networkx as nx
from gym import Env
from agents import CausalAgent, DiscreteAgentInterventionStructure
import numpy as np
from envs.generation.scm_gen import SCMGenerator, StructuralCausalModel
from episode_evals import EvalFunc
import copy
from pandas import DataFrame


class SCMEnvironment(Env):
    '''Defines a GYM environment over an SCM'''
    Agent: DiscreteAgentInterventionStructure
    Function = Callable[[], bool]

    def __init__(self, agent: CausalAgent,
                 episode_length: int,
                 scm: StructuralCausalModel,
                 mode: Union['train', 'evaluation']):
        super(SCMEnvironment, self).__init__()
        self.episode_length = episode_length
        self.current_action_indicies = [-1, -1]
        self.mode = mode

        # initialize causal model
        self.scm = scm
        self.var_values = self.scm.get_next_instantiation()[0]

        self.agent = agent
        self.action_space = self.agent.action_space
        self.observation_space = self.agent.observation_space
        self.prev_action = None
        self.old_samples = []

        self.steps_this_episode = 0

        # initialize observation vector
        # Sample part
        self.samples = np.array([[-10 for i in range(int((self.agent.observation_space.shape[0]-len(self.var_values)**2-1) / episode_length))]
                                 for _ in range(episode_length)], dtype=np.float32)
        # graph estimate part
        self.observation = np.pad(self.samples.flatten(), (0,len(self.var_values)**2+1), mode='constant', constant_values=-1)

        self.update_obs_vector()

        # create sample buffer that collects all samples and intervention targets for this environment across episodes
        self.samples_so_far = DataFrame(columns=list(self.scm.endogenous_vars.keys())+['INT'])

    def reset(self) -> np.ndarray:
        # initialize observation vector
        # Sample part
        self.samples = np.array([[-10 for i in range(
            int((self.agent.observation_space.shape[0] - len(self.var_values) ** 2 - 1) / self.episode_length))]
                                 for _ in range(self.episode_length)], dtype=np.float32)
        # graph estimate part
        self.observation = np.pad(self.samples.flatten(), (0, len(self.var_values) ** 2 + 1), mode='constant',
                                  constant_values=-1)

        self.steps_this_episode = 0
        self.agent.reset_causal_model(mode='empty')
        # reset observations
        self.old_samples = []
        self.update_obs_vector()

        self.prev_epi_graph = copy.deepcopy(self.agent.causal_model)
        return self.observation

    def get_observation(self):
        return self.observation

    def get_latest_action(self):
        return self.agent.current_action

    def get_causal_structure(self):
        return self.scm.create_graph()

    def get_epistemic_model(self):
        return self.agent.causal_model

    def calculate_reward(self) -> float:
        reward = 0
        gt_graph = self.scm.create_graph()

        # reward shaping every step
        last_cd_action = self.agent.current_action[1]
        shaping_reward = 0
        if last_cd_action:
            last_cd_action_upper = ((last_cd_action[0][0].upper(), last_cd_action[0][1].upper()),
                                    last_cd_action[1])
            # edge was deleted
            if last_cd_action[1] == -1 and last_cd_action[0] in self.prev_epi_graph.edges:
                # deleted an edge that is not in the ground truth
                if last_cd_action_upper[0] not in gt_graph.edges:
                    shaping_reward += 1
                # deleted an edge that is in the ground truth
                elif last_cd_action_upper[0] in gt_graph.edges:
                    shaping_reward -= 1

            # edge was added
            elif last_cd_action[1] == 1 and last_cd_action[0] not in self.prev_epi_graph.edges:
                # added an edge that is not in the ground truth
                if last_cd_action_upper[0] not in gt_graph.edges:
                    shaping_reward -= 1
                # added an edge that is in the ground truth
                elif last_cd_action_upper[0] in gt_graph.edges:
                    shaping_reward += 1

        # update shaped rewards
        reward = reward + shaping_reward
        return reward

    def map_act_to_var(self, int_act_index: int) -> str:
        return 'X' + str(int_act_index)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_action_indicies = action
        self.agent.current_action = self.agent.get_action_from_actionspace_sample(action)

        # apply action
        if not self.agent.current_action[0] is None:  # Intervention action
            self.scm.do_interventions([(self.map_act_to_var(self.agent.current_action[0][0]),
                                        lambda: self.agent.current_action[0][1])])

        if not self.agent.current_action[1] is None:  # Structure action
            self.agent.update_model_per_action(self.agent.current_action[1])

        # sample the environment's SCM
        self.var_values = self.scm.get_next_instantiation()[0]

        # store samples
        # make intervention target 1-based
        if self.agent.current_action[0]:
            int_index = self.agent.current_action[0][0] + 1
        else:
            int_index = 0
        self.samples_so_far.loc[len(self.samples_so_far)+1] = self.var_values + [int(int_index)]

        if not self.agent.current_action[0] is None:  # intervention was done
            self.scm.undo_interventions()

        # Determine the reward
        if self.mode == 'train':
            reward = self.calculate_reward()
        else:
            reward = 0

        self.prev_epi_graph = copy.deepcopy(self.agent.causal_model)
        self.steps_this_episode += 1

        # determine observation after action
        self.update_obs_vector()

        # evaluate the step
        done = self.steps_this_episode >= self.episode_length - 1

        self.prev_action = self.agent.current_action
        if done:
            # reset environment if episode is done
            self.reset()

        return self.observation, reward, done, {}

    def update_obs_vector(self):
        # compute vector for current observation
        intervention_one_hot = [0.0 for _ in range(self.agent.action_space[0].n)]
        intervention_one_hot[self.current_action_indicies[0]] = 1.0
        sample = [float(l) for l in self.var_values]
        sample.extend(intervention_one_hot)
        self.samples[self.steps_this_episode] = sample

        # get graph adjaciency
        graph_state = self.agent.get_graph_state()
        steps_left = (self.episode_length - self.steps_this_episode) / self.episode_length
        graph_state.append(steps_left)

        # update observation accordingly
        self.observation[:len(self.samples.flatten())] = self.samples.flatten()
        self.observation[len(self.samples.flatten()):] = np.array(graph_state).flatten()

    def render(self, mode: str = 'human') -> NoReturn:
        if mode == 'human':
            out = ''
            for i in range(len(self.var_values)):
                out += str(round(self.var_values[i], 3))
                if self.current_action_indicies[0] == i:
                    out += '*'
                out += '\t'
            print(out)


class SCMEnvironmentReservoir(Env):
    '''Defines a GYM environment over a set of SCMs'''
    envs: List[SCMEnvironment]

    def __init__(self, n_vars: int,
                 agent_type: type(CausalAgent),
                 eval_func_type: type(EvalFunc),
                 episode_length: int,
                 possible_functions: List[str],
                 test_set: Optional[List[nx.DiGraph]] = None,
                 train_set: Optional[List[nx.DiGraph]] = None,
                 intervention_value: int = 20,
                 mode: str = 'train'):

        self.envs = []
        self.test_set = test_set
        self.train_set = train_set
        self.eval_func_type = eval_func_type
        self.agent_type = agent_type
        self.episode_length = episode_length
        self.n_vars = n_vars
        self.possible_functions = possible_functions
        self.gen = SCMGenerator()
        self.intervention_value = intervention_value
        self.mode = mode

        self.current_env = self.get_next_env(new_env=True)
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def _build_agent_eval_func(self):
        if self.agent_type == DiscreteAgentInterventionStructure or self.agent_type == DiscreteAgentIntervention:
            agent = self.agent_type(self.n_vars, interv_value=self.intervention_value, state_repeats=self.episode_length)
        else:
            raise NotImplementedError('Agent type ' + str(self.agent_type) + 'is not implemented.')
        eval_func = None
        return agent, eval_func

    def reset(self):
        # reset the current environment
        self.current_env.reset()

        # choose a random next environment and reset it
        self.current_env = self.get_next_env(new_env=True)
        return self.current_env.reset()

    def get_observation(self):
        return self.current_env.observation

    def get_latest_action(self):
        return self.current_env.get_latest_action()

    def get_causal_structure(self):
        return self.current_env.get_causal_structure()

    def get_epistemic_model(self):
        return self.current_env.get_epistemic_model()

    def set_reward(self, reward: float):
        self.current_env.set_reward(reward)

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode='human'):
        self.current_env.render(mode)

    def get_next_env(self, new_env: bool = True):
        """
        Samples a new environment in the reservoir if new_env is True. The environment is only returned if its graph
        structure is not in the test.

        :param new_env: If True, a new environment is samples. If False, the environment of the previous episode
        is returned
        """
        if not self.train_set is None:  # sample new environment from the train set if exists
            graph = random.choice(self.train_set)
            scm = self.gen.create_scm_from_graph(graph, possible_functions=[k for k in self.possible_functions])

        elif not self.test_set is None and self.train_set is None:  # if only test set is provided, sample a scm that
            # is not in the test set

            while True:  # resample scm until there is one that is not in the test set
                scm = self.gen.create_random(possible_functions=[k for k in self.possible_functions],
                                             n_endo=self.n_vars, n_exo=0)[0]
                graph = scm.create_graph()

                in_testset = False
                for i in range(len(self.test_set)):
                    edit_distance = nx.graph_edit_distance(self.test_set[i], graph)
                    if edit_distance == 0:
                        in_testset = True
                        break
                if not in_testset:
                    break

        else:  # self.test_set is None and self.train_set is None:
            raise RuntimeError('A test set must be provided')

        agent, _ = self._build_agent_eval_func()
        return SCMEnvironment(agent, self.episode_length, scm, mode=self.mode)

