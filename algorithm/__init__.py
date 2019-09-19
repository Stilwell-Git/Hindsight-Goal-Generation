from .ddpg import DDPG

def create_agent(args):
	return {
		'ddpg': DDPG
	}[args.alg](args)