"""Training script to train an MCD policy"""

import argparse
import dill
import copy

from envs.environments import SCMEnvironmentReservoir
from envs.callbacks import EvalTrainTestCallback
from agents import DiscreteAgentInterventionStructure
from episode_evals import FixedLengthStructEpisode
from envs.generation.scm_gen import SCMGenerator


from envs.callbacks import CheckpointCallback
from stable_baselines3.common import vec_env
from gym.spaces.multi_discrete import MultiDiscrete
from cddqn import CDDQN, QNetwork
from dualDQN import DualDQN
from DQNrandINT import DQNrandINT

import wandb


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='data/', help='Filepath of where to save the data.')
    parser.add_argument('--total-steps', type=int, default=5000000, help='Total amount of steps to train the model.')
    parser.add_argument('--ep-length', type=int, default=15, help='Episode length.')
    parser.add_argument('--test-set', type=str, default='data/3var_all_graphs/test.pkl', help='Path to pickled file with testing data.')
    parser.add_argument('--train-set', type=str, default='data/3var_all_graphs/train.pkl',
                        help='Path to pickled file with training data.')
    parser.add_argument('--possible-functions', nargs='+', default=['xor'],
                        help='list of function types that are allowed when generating the SCMs for '
                             'testing and training.')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers to run in parallel.')
    parser.add_argument('--val-frequency', type=int, default=5000,
                        help='Frequency in training steps in which the agent should be evaluated.')
    parser.add_argument('--load-model-path', default=None, type=str, help='Path to load a pretrained model from.'
                                                                          ' \'None\' if it should be trained from scratch.')
    parser.add_argument('--n-eval-episodes', type=int, default=20, help='How many episodes should be done for '
                                                                        'each evaluation.')
    parser.add_argument('--tags', nargs='+', default=["long", "3vars", "all_graphs", "cluster"],
                        help='List of tags for the current weights and biases session.')
    parser.add_argument('--layer-size', type=int, default=128, help='Size of each hidden layer')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of hidden layers.')
    parser.add_argument('--lr', type=float, default=0.00001, help='The learning rate.')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor.')
    parser.add_argument('--interv-value', type=int, default=20, help='The value of variables on which an intervention '
                                                                     'is performed.')
    # DQN arguments
    parser.add_argument('--learning-starts', type=int, default=50000,
                        help='After how many steps DQN should start learning')
    parser.add_argument('--exploration-fraction', type=float, default=0.7, help='Fraction over entire training period '
                                                                                'over which the exploration rate is reduced.')
    parser.add_argument('--initial-eps', type=float, default=1.0, help='Initial exploration rate.')
    parser.add_argument('--final-eps', type=float, default=0.1, help='Final exploration rate.')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Maximum number of sample is the replay buffer.')

    return parser.parse_args()


def train(args):
    #setup wandb
    config = vars(args)
    wandb.init(
        entity='',
        project='',
        tags=args.tags,
        config=config
    )
    algo = 'DQN'

    # load data
    if not args.test_set is None:
        with open(args.test_set, "rb") as f:
            test_dags = dill.load(f)
    if not args.train_set is None:
        with open(args.train_set, "rb") as f:
            train_dags = dill.load(f)
    else:
        train_dags = None

    n_vars = len(test_dags[0].nodes)

    # create the environment
    env = SCMEnvironmentReservoir(n_vars, DiscreteAgentInterventionStructure, FixedLengthStructEpisode,
                                  args.ep_length, possible_functions=args.possible_functions, test_set=test_dags,
                                  train_set=train_dags, intervention_value=args.interv_value)
    # create vectorized environments for parallelization
    env = vec_env.SubprocVecEnv([lambda: copy.deepcopy(env) for _ in range(args.workers)])
    device = 'cuda'

    if algo == 'DQNrandINT':
        model = DQNrandINT(policy=CDDQN, learning_starts=args.learning_starts, env=env, buffer_size=args.buffer_size,
                           policy_kwargs={'net_arch': [args.layer_size for _ in range(args.n_layers)],
                           'nr_nodes': n_vars, 'obs_dim': env.observation_space.shape[0]},
                           device=device, exploration_fraction=args.exploration_fraction,
                           exploration_initial_eps=args.initial_eps, exploration_final_eps=args.final_eps,
                           learning_rate=args.lr, gamma=args.gamma, batch_size=32768)

    elif algo == 'DQN':
        # load pretrained model is specified
        if args.load_model_path:
            # load model
            pi_in = QNetwork.load(args.load_model_path + '/best_model_in')
            pi_cd = QNetwork.load(args.load_model_path + '/best_model_cd')
            action_space = MultiDiscrete([pi_in.action_space.n, pi_cd.action_space.n])
            policy = CDDQN(pi_in.observation_space, action_space, lr_schedule=lambda x: x, nr_nodes=n_vars,
                           obs_dim=pi_in.observation_space.shape)
            policy.q_net_cd = pi_cd
            policy.q_net_cd_target = pi_cd
            policy.q_net_in = pi_in
            policy.q_net_in_target = pi_in
            # Setup optimizer with initial learning rate
            policy.optimizer_cd = policy.optimizer_class(policy.q_net_cd.parameters(), lr=args.lr)
            policy.optimizer_in = policy.optimizer_class(policy.q_net_in.parameters(), lr=args.lr)

            model = DualDQN(policy=CDDQN, learning_starts=args.learning_starts, env=env, buffer_size=args.buffer_size,
                            policy_kwargs={'net_arch': [64], 'nr_nodes': n_vars,
                                           'obs_dim': policy.obs_dim},
                            device=device, exploration_fraction=args.exploration_fraction,
                            exploration_initial_eps=args.initial_eps, exploration_final_eps=args.final_eps,
                            learning_rate=args.lr, gamma=args.gamma, batch_size=32768)
            model.policy = policy
            model._create_aliases()

        # Create new model if not specified
        else:
            model = DualDQN(policy=CDDQN, learning_starts=args.learning_starts, env=env, buffer_size=args.buffer_size,
                            policy_kwargs={'net_arch': [args.layer_size for _ in range(args.n_layers)],
                                           'nr_nodes': n_vars, 'obs_dim': env.observation_space.shape[0]},
                            device=device, exploration_fraction=args.exploration_fraction,
                            exploration_initial_eps=args.initial_eps, exploration_final_eps=args.final_eps,
                            learning_rate=args.lr, gamma=args.gamma, batch_size=32768)

    # setup callbacks
    checkpoint_cb = CheckpointCallback(save_freq=int(args.val_frequency / args.workers),  # args.workers
                                       save_path=args.save_dir,
                                       name_prefix='latest_model')

    # Create SCMs from test DAGs
    test_scms = [SCMGenerator().create_scm_from_graph(graph, args.possible_functions) for graph in test_dags]  # 1 SCM per dag
    eval_cb = EvalTrainTestCallback(val_frequency=int(args.val_frequency / args.workers),
                                    scms_val=test_scms,
                                    n_vars=n_vars,
                                    episode_length=args.ep_length,
                                    best_model_save_path=args.save_dir,
                                    intervention_value=args.interv_value)
    # main training loop
    model.learn(args.total_steps, callback=[checkpoint_cb, eval_cb])

    env.close()
    del env
    del model


if __name__ == '__main__':
    args = get_params()
    train(args)

