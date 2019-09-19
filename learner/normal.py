import numpy as np
from envs import make_env
from algorithm.replay_buffer import Trajectory

class NormalLearner:
	def __init__(self, args):
		pass

	def learn(self, args, env, env_test, agent, buffer):
		for _ in range(args.episodes):
			obs = env.reset()
			current = Trajectory(obs)
			for timestep in range(args.timesteps):
				action = agent.step(obs, explore=True)
				obs, reward, done, _ = env.step(action)
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()

		for _ in range(args.test_rollouts):
			def test_rollout(env, prefix=''):
				rewards = 0.0
				obs = env.reset()
				for timestep in range(args.timesteps):
					action, info = agent.step(obs, explore=False, test_info=True)
					args.logger.add_dict(info, prefix)
					obs, reward, done, info = env.step(action)
					rewards += reward
					if timestep==args.timesteps-1: done = True
					if done: break
				args.logger.add_dict(info, prefix)

			if args.goal_based:
				# goal-based envs test
				test_rollout(env, 'train/')
				test_rollout(env_test, 'test/')
			else:
				# normal envs test
				test_rollout(env)