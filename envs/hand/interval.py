import gym
import numpy as np
from .fixobj import FixedObjectGoalEnv
from envs.utils import quat_from_angle_and_axis

class IntervalGoalEnv(FixedObjectGoalEnv):
	def __init__(self, args):
		FixedObjectGoalEnv.__init__(self, args)

	def generate_goal(self):
		# Select a goal for the object position.
		target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]

		# Select a goal for the object rotation.
		if self.args.env!='HandManipulatePen-v0':
			angle = np.pi + (np.random.uniform(-1.0,1.0))*(np.pi/4.0)
		else:
			angle = np.pi/2.0 + (np.random.uniform(-1.0,1.0))*(np.pi/4.0)
		axis = np.array([0., 0., 1.])
		target_quat = quat_from_angle_and_axis(angle, axis)

		target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
		goal = np.concatenate([target_pos, target_quat])
		return goal.copy()