from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .obstacle import ObstacleGoalEnv

def make_env(args):
	return {
		'vanilla': VanillaGoalEnv,
		'fixobj': FixedObjectGoalEnv,
		'interval': IntervalGoalEnv,
		'obstacle': ObstacleGoalEnv
	}[args.goal](args)
