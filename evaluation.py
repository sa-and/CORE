"""Script for running and evaluating a MCD policy"""

from envs.environments import SCMEnvironment
from envs.generation.scm_gen import SCMGenerator, StructuralCausalModel
from envs.generation.graph_gen import CausalGraphGenerator
from agents import DiscreteAgentInterventionStructure
import networkx as nx
import numpy as np
from episode_evals import directed_shd
from typing import Union
from time import perf_counter
import wandb
from cddqn import QNetwork, CDDQN
from dualDQN import DualDQN
from gym.spaces.multi_discrete import MultiDiscrete
import dill
import tqdm


def apply_policy(model: DualDQN, test_env: Union[StructuralCausalModel, SCMEnvironment], n_vars, episode_length,
                 intervention_value, logging):
    """
    Applies the given model to the test_env for one episode and returns the produced graph.
    """
    model_workers = model.n_envs
    if type(test_env) == StructuralCausalModel:
        test_env = SCMEnvironment(agent=DiscreteAgentInterventionStructure(n_vars, state_repeats=episode_length,
                                                                           interv_value=intervention_value),
                                  scm=test_env,
                                  episode_length=episode_length,
                                  mode='evaluation')

    episode_starts = np.ones((model_workers,), dtype=bool)
    obs = test_env.reset()
    obs = np.array([obs for _ in range(model_workers)])
    for i in range(episode_length-1):
        rewards = []
        actions, _ = model.predict(obs, episode_start=episode_starts, deterministic=True)
        obs, reward, done, _ = test_env.step(actions[0])
        rewards.append(reward)
        # copy obs and dones for all workers
        obs = np.array([obs for _ in range(model_workers)])
        done = [done for _ in range(model_workers)]
        episode_starts = done

    if logging:
        wandb.log({'eval/final_episode_reward': rewards[-1],
                   'eval/mean_episode_reward': sum(rewards)/len(rewards)}, commit=False)

    return nx.DiGraph(test_env.agent.causal_model)


def evaluate_policy(model: Union[str, DualDQN], eval_data, runs_per_env: int, n_vars: int, episode_length: int,
                    logging: bool = True, intervention_value: int = 20) -> np.array:
    """
    Applies the given policy runs_per_env times on each of the environments givne in the eval_data.
    """
    if type(model) == str:
        raise NotImplementedError('can\'t load from path.')

    final_rewards = []
    times = []
    edges = []
    for scm in tqdm.tqdm(eval_data):
        target_graph = scm.create_graph()

        for run in range(runs_per_env):
            start = perf_counter()
            predicted_graph = apply_policy(  model=model,
                                             test_env=scm,
                                             n_vars=n_vars,
                                             episode_length=episode_length,
                                             intervention_value=intervention_value,
                                             logging=logging)
            end = perf_counter()

            difference = directed_shd(predicted_graph, target_graph)
            # print(difference)
            edges.append(len(predicted_graph.edges))
            final_rewards.append(difference)
            times.append(end-start)
            if logging:
                wandb.log({'eval/execution_time': end-start,
                           'eval/final_shd': difference}, commit=False)

    final_rewards = np.array(final_rewards)
    times = np.array(times)
    edges = np.array(edges)

    return final_rewards, times, edges


if __name__ == '__main__':
    algo = 'dualdqn'  # 'random'  # 'empty'
    path = 'exp/5var/lin_nonoise_20/ours/'
    in_model = 'best_model_in'
    cd_model = 'best_model_cd'
    test_set_path = 'data/5var_16000_sparse/test.pkl'
    possible_functions = ['linear']
    episode_length = 10
    runs = 1
    vars = 5
    interv_value = 20

    # load test data
    with open(test_set_path, "rb") as f:
        test_dags = dill.load(f)

    gen = CausalGraphGenerator(vars)
    diffs = []
    edges = []

    if algo == 'random':
        for run in range(runs):
            for dag in test_dags:
                if vars <= 4:  # not er graphs
                    graph = gen.generate_random_graph(method='full')[0]
                else:  # ER graphs
                    graph = gen.generate_random_graph(p=0.2)[0]
                diffs.append(directed_shd(graph, dag))
                edges.append(len(graph.edges))

    elif algo == 'empty':
        for run in range(runs):
            for dag in test_dags:
                predicted_graph = gen.generate_random_graph(p=0.0)[0]
                diffs.append(directed_shd(predicted_graph, dag))
                edges.append(len(predicted_graph.edges))

    elif algo == 'dualdqn':
        test_scms = [SCMGenerator().create_scm_from_graph(test_dags[i % len(test_dags)], possible_functions=possible_functions) for i in
         range(len(test_dags[:999] * runs))]
        # # create the environment
        dummy_env = SCMEnvironment(DiscreteAgentInterventionStructure(vars, state_repeats=episode_length),
                                           episode_length, test_scms[0], mode='evaluation')
        # load model
        pi_in = QNetwork.load(path+in_model)
        pi_cd = QNetwork.load(path+cd_model)
        action_space = MultiDiscrete([pi_in.action_space.n, pi_cd.action_space.n])
        policy = CDDQN(pi_in.observation_space, action_space, lr_schedule=lambda x: x, nr_nodes=vars,
                       obs_dim=pi_in.observation_space.shape)
        policy.q_net_cd = pi_cd
        policy.q_net_in = pi_in

        model = DualDQN(policy=CDDQN, learning_starts=0, env=dummy_env, buffer_size=50000,
                        policy_kwargs={'net_arch': [64], 'nr_nodes': len(test_dags[0].nodes), 'obs_dim': policy.obs_dim},
                        device='cpu')
        model.policy = policy
        try:
            diff, times, edge = evaluate_policy(model=model, eval_data=test_scms, runs_per_env=runs, n_vars=vars,
                                                episode_length=episode_length, logging=False, intervention_value=interv_value)
            diffs.append(diff.tolist())
            edges.append(edge.tolist())
        except ValueError as e:
            raise ValueError(e.args[0] + ' Probably the episode length doesn\'t match the episode length that the '
                                         'model was trained on.')

    diffs = np.array(diffs)
    edges = np.array(edges)
    print('mean diffs:', diffs.mean())
    print('std diffs:', diffs.std())
    print('mean n edges', edges.mean())

