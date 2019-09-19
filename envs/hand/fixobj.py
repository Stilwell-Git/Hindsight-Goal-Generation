import gym
import numpy as np
from .vanilla import VanillaGoalEnv
from gym.envs.robotics import rotations
from envs.utils import quat_from_angle_and_axis

class FixedObjectGoalEnv(VanillaGoalEnv):
	def __init__(self, args):
		VanillaGoalEnv.__init__(self, args)
		self.env.reset()
		self.fixed_obj = True

	def reset(self):
		self.reset_ep()
		self.sim.set_state(self.initial_state)
		self.sim.forward()

		initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
		initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
		assert initial_qpos.shape == (7,)
		assert initial_pos.shape == (3,)
		assert initial_quat.shape == (4,)
		initial_qpos = None

		angle = self.np_random.uniform(-np.pi, np.pi)*self.args.init_rotation
		axis = np.array([0., 0., 1.])
		offset_quat = quat_from_angle_and_axis(angle, axis)
		initial_quat = rotations.quat_mul(initial_quat, offset_quat)

		initial_quat /= np.linalg.norm(initial_quat)
		initial_qpos = np.concatenate([initial_pos, initial_quat])
		self.sim.data.set_joint_qpos('object:joint', initial_qpos)

		def is_on_palm():
			self.sim.forward()
			cube_middle_idx = self.sim.model.site_name2id('object:center')
			cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
			is_on_palm = (cube_middle_pos[2] > 0.04)
			return is_on_palm

		# Run the simulation for a bunch of timesteps to let everything settle in.
		for _ in range(10):
			self._set_action(np.zeros(20))
			try:
				self.sim.step()
			except mujoco_py.MujocoException:
				return False
		assert is_on_palm()

		self.goal = self.generate_goal()
		self.last_obs = (self.get_obs()).copy()
		return self.get_obs()

	def generate_goal(self):
		return self.env.env._sample_goal()