import math
import numpy as np
from gym.envs.robotics.hand.manipulate import quat_from_angle_and_axis

def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)

def fetchpush_goal_projection(goal):
	x, y, z = goal[0], goal[1], 0.425
	if x>1.45: x = 1.45
	if y<0.6: y = 0.6
	if y>0.9: y = 0.9
	if min(abs(y-0.6),abs(y-0.9))<abs(x-1.45):
		y = 0.6 if y<0.75 else 0.9
	else:
		x = 1.45

	goal_proj = np.array([x,y,z])
	if x<1.45:
		if y<0.75:
			return -(1.45-x), goal_proj
		else:
			return 0.3+(1.45-x), goal_proj
	else:
		return y-0.6, goal_proj

def fetchpush_obstacle_distance(goal_a, goal_b):
	a, g_a = fetchpush_goal_projection(goal_a)
	b, g_b = fetchpush_goal_projection(goal_b)
	return abs(a-b)+goal_distance(goal_a,g_a)+goal_distance(goal_b,g_b)

def get_goal_distance(args):
	if args.goal=='obstacle':
		assert args.env=='FetchPush-v1'
		return fetchpush_obstacle_distance
	else:
		return goal_distance

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
