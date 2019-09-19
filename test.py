import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir

class Tester:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)

		self.info = []
		if args.save_acc:
			make_dir('log/accs', clear=False)
			self.test_rollouts = 100

			self.env_List = []
			self.env_test_List = []
			for _ in range(self.test_rollouts):
				self.env_List.append(make_env(args))
				self.env_test_List.append(make_env(args))

			self.acc_record = {}
			self.acc_record[self.args.goal] = []
			for key in self.acc_record.keys():
				self.info.append('Success/'+key+'@blue')

	def test_acc(self, key, env, agent):
		acc_sum, obs = 0.0, []
		for i in range(self.test_rollouts):
			obs.append(goal_based_process(env[i].reset()))
		for timestep in range(self.args.timesteps):
			actions = agent.step_batch(obs)
			obs, infos = [], []
			for i in range(self.test_rollouts):
				ob, _, _, info = env[i].step(actions[i])
				obs.append(goal_based_process(ob))
				infos.append(info)
		for i in range(self.test_rollouts):
			acc_sum += infos[i]['Success']

		steps = self.args.buffer.counter
		acc = acc_sum/self.test_rollouts
		self.acc_record[key].append((steps,acc))
		self.args.logger.add_record('Success/'+key, acc)

	def cycle_summary(self):
		if self.args.save_acc:
			self.test_acc(self.args.goal, self.env_List, self.args.agent)

	def epoch_summary(self):
		if self.args.save_acc:
			for key, acc_info in self.acc_record.items():
				log_folder = 'accs'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)

	def final_summary(self):
		if self.args.save_acc:
			for key, acc_info in self.acc_record.items():
				log_folder = 'accs'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)