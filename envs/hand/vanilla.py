import gym
import copy
import numpy as np
from envs.utils import goal_distance, goal_distance_obs
from utils.os_utils import remove_color

class VanillaGoalEnv():
	def __init__(self, args):
		self.args = args
		rotate_env_id = {
			'HandManipulateBlock-v0': 'HandManipulateBlockRotateXYZ-v0',
			'HandManipulateEgg-v0': 'HandManipulateEggRotate-v0',
			'HandManipulatePen-v0': 'HandManipulatePenRotate-v0',
			'HandReach-v0': 'HandReach-v0'
		}[args.env]
		self.env = gym.make(rotate_env_id)
		self.np_random = self.env.env.np_random

		if self.args.env=='HandReach-v0':
			self.distance_threshold = self.env.env.distance_threshold
		else:
			self.distance_threshold = self.env.env.rotation_threshold/4.0 # just for estimation
			assert self.env.env.target_position=='ignore'
			assert self.env.env.target_rotation!='ignore'

		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.max_episode_steps = self.env._max_episode_steps

		self.fixed_obj = False

		self.render = self.env.render
		self.reset_sim = self.env.env._reset_sim
		self._set_action = self.env.env._set_action

		self.reset()
		if self.args.env=='HandReach-v0':
			self.env_info = {
				'Rewards': self.process_info_rewards, # episode cumulative rewards
				'Distance': self.process_info_distance, # estimated distance in the last step
				'Success@green': self.process_info_success # is_success in the last step
			}
		else:
			self.env_info = {
				'Rewards': self.process_info_rewards, # episode cumulative rewards
				'Distance': self.process_info_distance, # estimated distance in the last step
				'Rotation': self.process_info_rotation, # groundtruth distance in the last step
				'Success@green': self.process_info_success # is_success in the last step
			}

	def compute_reward(self, achieved_goal, goal):
		if self.args.env=='HandReach-v0':
			return self.env.env.compute_reward(achieved_goal[0], goal, None)
		else:
			goal_a = np.concatenate([np.zeros(3), achieved_goal[0]], axis=0)
			goal_b = np.concatenate([np.zeros(3), goal], axis=0)
			return self.env.env.compute_reward(goal_a, goal_b, None)

	def compute_distance(self, achieved, goal):
		return np.sqrt(np.sum(np.square(achieved-goal)))

	def process_info_rewards(self, obs, reward, info):
		self.rewards += reward
		return self.rewards

	def process_info_distance(self, obs, reward, info):
		# different with internal reward
		return self.compute_distance(obs['achieved_goal'], obs['desired_goal'])

	def process_info_rotation(self, obs, reward, info):
		_, d_rot = self.env.env._goal_distance(obs['achieved_goal'], obs['desired_goal'])
		return d_rot

	def process_info_success(self, obs, reward, info):
		return info['is_success']

	def process_info(self, obs, reward, info):
		return {
			remove_color(key): value_func(obs, reward, info)
			for key, value_func in self.env_info.items()
		}

	def rotate_obs(self, obs):
		return copy.deepcopy({
			'observation': obs['observation'],
			'desired_goal': obs['desired_goal'][3:],
			'achieved_goal': obs['achieved_goal'][3:]
		})

	def get_obs(self):
		if self.args.env=='HandReach-v0':
			return self.env.env._get_obs()
		else:
			return self.rotate_obs(self.env.env._get_obs())

	def step(self, action):
		# imaginary infinity horizon (without done signal)
		raw_obs, reward, done, info = self.env.step(action)
		info = self.process_info(raw_obs, reward, info)
		obs = self.get_obs()
		self.last_obs = obs.copy()
		return obs, reward, False, info

	def reset_ep(self):
		self.rewards = 0.0

	def reset(self):
		self.reset_ep()
		self.env.reset()
		self.last_obs = (self.get_obs()).copy()
		return self.last_obs.copy()

	def set_target(self, target_id, target_pos):
		sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
		site_id = self.sim.model.site_name2id('target'+str(target_id))
		self.sim.model.site_pos[site_id] = target_pos - sites_offset[0]
		self.sim.forward()

	@property
	def sim(self):
		return self.env.env.sim
	@sim.setter
	def sim(self, new_sim):
		self.env.env.sim = new_sim

	@property
	def initial_state(self):
		return self.env.env.initial_state

	@property
	def initial_gripper_xpos(self):
		return self.env.env.initial_gripper_xpos.copy()

	@property
	def goal(self):
		return self.env.env.goal.copy()
	@goal.setter
	def goal(self, value):
		if self.args.env=='HandReach-v0':
			self.env.env.goal = value.copy()
		else:
			if np.prod(value.shape)==4:
				target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]
				value = np.concatenate([target_pos, value])
			self.env.env.goal = value.copy()
