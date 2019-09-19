import numpy as np
import copy
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import get_arg_parser, get_logger, str2bool
from algorithm import create_agent
from learner import create_learner, learner_collection
from test import Tester
from algorithm.replay_buffer import ReplayBuffer_Episodic, goal_based_process

def get_args():
	parser = get_arg_parser()

	parser.add_argument('--tag', help='terminal tag in logger', type=str, default='')
	parser.add_argument('--alg', help='backend algorithm', type=str, default='ddpg', choices=['ddpg', 'ddpg2'])
	parser.add_argument('--learn', help='type of training method', type=str, default='hgg', choices=learner_collection.keys())

	parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1', choices=Robotics_envs_id)
	args, _ = parser.parse_known_args()
	if args.env=='HandReach-v0':
		parser.add_argument('--goal', help='method of goal generation', type=str, default='reach', choices=['vanilla', 'reach'])
	else:
		parser.add_argument('--goal', help='method of goal generation', type=str, default='interval', choices=['vanilla', 'fixobj', 'interval'])
		if args.env[:5]=='Fetch':
			parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32, default=1.0)
		elif args.env[:4]=='Hand':
			parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32, default=0.25)

	parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.98)
	parser.add_argument('--clip_return', help='whether to clip return value', type=str2bool, default=True)
	parser.add_argument('--eps_act', help='percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
	parser.add_argument('--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32, default=0.2)

	parser.add_argument('--pi_lr', help='learning rate of policy network', type=np.float32, default=1e-3)
	parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=1e-3)
	parser.add_argument('--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
	parser.add_argument('--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32, default=0.95)

	parser.add_argument('--epoches', help='number of epoches', type=np.int32, default=20)
	parser.add_argument('--cycles', help='number of cycles per epoch', type=np.int32, default=20)
	parser.add_argument('--episodes', help='number of episodes per cycle', type=np.int32, default=50)
	parser.add_argument('--timesteps', help='number of timesteps per episode', type=np.int32, default=(50 if args.env[:5]=='Fetch' else 100))
	parser.add_argument('--train_batches', help='number of batches to train per cycle', type=np.int32, default=20)

	parser.add_argument('--buffer_size', help='number of episodes in replay buffer', type=np.int32, default=10000)
	parser.add_argument('--buffer_type', help='type of replay buffer / whether to use Energy-Based Prioritization', type=str, default='energy', choices=['normal','energy'])
	parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=256)
	parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)
	parser.add_argument('--her', help='type of hindsight experience replay', type=str, default='future', choices=['none', 'final', 'future'])
	parser.add_argument('--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
	parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full', choices=['full', 'final'])

	parser.add_argument('--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
	parser.add_argument('--hgg_L', help='Lipschitz constant', type=np.float32, default=5.0)
	parser.add_argument('--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)

	parser.add_argument('--save_acc', help='save successful rate', type=str2bool, default=True)

	args = parser.parse_args()
	args.goal_based = (args.env in Robotics_envs_id)
	args.clip_return_l, args.clip_return_r = clip_return_range(args)

	logger_name = args.alg+'-'+args.env+'-'+args.learn
	if args.tag!='': logger_name = args.tag+'-'+logger_name
	args.logger = get_logger(logger_name)

	for key, value in args.__dict__.items():
		if key!='logger':
			args.logger.info('{}: {}'.format(key,value))

	return args

def experiment_setup(args):
	env = make_env(args)
	env_test = make_env(args)
	if args.goal_based:
		args.obs_dims = list(goal_based_process(env.reset()).shape)
		args.acts_dims = [env.action_space.shape[0]]
		args.compute_reward = env.compute_reward
		args.compute_distance = env.compute_distance

	args.buffer = buffer = ReplayBuffer_Episodic(args)
	args.learner = learner = create_learner(args)
	args.agent = agent = create_agent(args)
	args.logger.info('*** network initialization complete ***')
	args.tester = tester = Tester(args)
	args.logger.info('*** tester initialization complete ***')

	return env, env_test, agent, buffer, learner, tester