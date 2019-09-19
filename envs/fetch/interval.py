import gym
import numpy as np
from .fixobj import FixedObjectGoalEnv

class IntervalGoalEnv(FixedObjectGoalEnv):
	def __init__(self, args):
		FixedObjectGoalEnv.__init__(self, args)

	def generate_goal(self):
		if self.has_object:
			goal = self.initial_gripper_xpos[:3] + self.target_offset
			if self.args.env=='FetchSlide-v1':
				goal[0] += self.target_range*0.5
				goal[1] += np.random.uniform(-self.target_range, self.target_range)*0.5
			else:
				goal[0] += np.random.uniform(-self.target_range, self.target_range)
				goal[1] += self.target_range
			goal[2] = self.height_offset + int(self.target_in_the_air)*0.45
		else:
			goal = self.initial_gripper_xpos[:3] + np.array([np.random.uniform(-self.target_range, self.target_range), self.target_range, self.target_range])
		return goal.copy()