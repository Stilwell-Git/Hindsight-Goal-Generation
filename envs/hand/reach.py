import gym
import numpy as np
from .vanilla import VanillaGoalEnv
from envs.utils import quat_from_angle_and_axis

class ReachGoalEnv(VanillaGoalEnv):
	def __init__(self, args):
		VanillaGoalEnv.__init__(self, args)

	def generate_goal(self):
		thumb_name = 'robot0:S_thtip'
		finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
		finger_name = self.np_random.choice(finger_names)

		thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
		finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
		assert thumb_idx != finger_idx

		# Pick a meeting point above the hand.
		meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
		# meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)
		meeting_pos[np.random.randint(meeting_pos.shape[0])] += 0.005

		# Slightly move meeting goal towards the respective finger to avoid that they
		# overlap.
		goal = self.initial_goal.copy().reshape(-1, 3)
		for idx in [thumb_idx, finger_idx]:
			offset_direction = (meeting_pos - goal[idx])
			offset_direction /= np.linalg.norm(offset_direction)
			goal[idx] = meeting_pos - 0.005 * offset_direction

		if self.np_random.uniform() < 0.1:
			# With some probability, ask all fingers to move back to the origin.
			# This avoids that the thumb constantly stays near the goal position already.
			goal = self.initial_goal.copy()
		return goal.flatten()