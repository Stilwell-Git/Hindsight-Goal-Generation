import gym
import numpy as np
import os
from .vanilla import VanillaGoalEnv
from gym.envs.robotics.fetch_env import FetchEnv
from gym.wrappers.time_limit import TimeLimit

class ObstacleGoalEnv(VanillaGoalEnv):
	def __init__(self, args):
		VanillaGoalEnv.__init__(self, args)
		env_id = {
			'FetchPush-v1': 'push'
		}
		assert args.env in env_id.keys()
		MODEL_XML_PATH = os.path.abspath('.')+'/envs/assets/fetch/'+env_id[args.env]+'_obstacle.xml'

		if env_id[args.env] in ['push']:
			initial_qpos = {
				'robot0:slide0': 0.405,
				'robot0:slide1': 0.48,
				'robot0:slide2': 0.0,
				'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
			}
			self.env = FetchEnv(
				MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
				gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
				obj_range=0.15, target_range=0.15, distance_threshold=0.05,
				initial_qpos=initial_qpos, reward_type='sparse')

		self.env = TimeLimit(self.env, max_episode_steps=args.timesteps) # A default wrapper of gym.

		self.render = self.env.render
		self.get_obs = self.env.env._get_obs
		self.reset_sim = self.env.env._reset_sim

		self.env.reset()
		self.reset()

	def reset(self):
		self.reset_ep()
		self.sim.set_state(self.initial_state)

		if self.has_object:
			object_xpos = self.initial_gripper_xpos[:2].copy()
			random_offset = np.random.uniform(0.3,1.0)*self.obj_range*self.args.init_offset
			object_xpos -= np.array([random_offset, self.obj_range])
			object_qpos = self.sim.data.get_joint_qpos('object0:joint')
			assert object_qpos.shape == (7,)
			object_qpos[:2] = object_xpos
			self.sim.data.set_joint_qpos('object0:joint', object_qpos)

		self.sim.forward()
		self.goal = self.generate_goal()
		self.last_obs = (self.get_obs()).copy()
		return self.get_obs()

	def generate_goal(self):
		return self.env.env._sample_goal()

	def generate_goal(self):
		if self.has_object:
			goal = self.initial_gripper_xpos[:3] + self.target_offset
			goal[0] += np.random.uniform(-self.target_range, -self.target_range*0.3)
			goal[1] += self.target_range
			goal[2] = self.height_offset + int(self.target_in_the_air)*0.45
		else:
			goal = self.initial_gripper_xpos[:3] + np.array([np.random.uniform(-self.target_range, self.target_range), self.target_range, self.target_range])
		return goal.copy()
