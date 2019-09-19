from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv

def make_env(args):
	return {
		'vanilla': VanillaGoalEnv,
		'fixobj': FixedObjectGoalEnv,
		'interval': IntervalGoalEnv
	}[args.goal](args)