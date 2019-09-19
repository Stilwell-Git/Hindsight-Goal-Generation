import math
import numpy as np
from gym.envs.robotics.hand.manipulate import quat_from_angle_and_axis

def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)

def goal_distance_obs(obs):
	return goal_distance(obs['achieved_goal'], obs['desired_goal'])

def quaternion_to_euler_angle(array):
	# from "Energy-Based Hindsight Experience Prioritization"
	w = array[0]
	x = array[1]
	y = array[2]
	z = array[3]
	ysqr = y * y
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.atan2(t0, t1)
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.asin(t2)
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.atan2(t3, t4)
	result = np.array([X, Y, Z])
	return result