from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .reach import ReachGoalEnv

def make_env(args):
	return {
		'vanilla': VanillaGoalEnv,
		'fixobj': FixedObjectGoalEnv,
		'interval': IntervalGoalEnv,
		'reach': ReachGoalEnv
	}[args.goal](args)