import numpy as np
import tensorflow as tf
from envs import goal_distance_obs
from utils.tf_utils import get_vars, Normalizer
from algorithm.replay_buffer import goal_based_process

class DDPG:
	def __init__(self, args):
		self.args = args
		self.create_model()

		self.train_info_pi = {
			'Pi_q_loss': self.pi_q_loss,
			'Pi_l2_loss': self.pi_l2_loss
		}
		self.train_info_q = {
			'Q_loss': self.q_loss
		}
		self.train_info = {**self.train_info_pi, **self.train_info_q}

		self.step_info = {
			'Q_average': self.q_pi
		}

	def create_model(self):
		def create_session():
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=config)

		def create_inputs():
			self.raw_obs_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
			self.raw_obs_next_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
			self.acts_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)
			self.rews_ph = tf.placeholder(tf.float32, [None, 1])

		def create_normalizer():
			with tf.variable_scope('normalizer'):
				self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
			self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
			self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)

		def create_network():
			def mlp_policy(obs_ph):
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					pi_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='pi_dense1')
					pi_dense2 = tf.layers.dense(pi_dense1, 256, activation=tf.nn.relu, name='pi_dense2')
					pi_dense3 = tf.layers.dense(pi_dense2, 256, activation=tf.nn.relu, name='pi_dense3')
					pi = tf.layers.dense(pi_dense3, self.args.acts_dims[0], activation=tf.nn.tanh, name='pi')
				return pi

			def mlp_value(obs_ph, acts_ph):
				state_ph = tf.concat([obs_ph, acts_ph], axis=1)
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					q_dense1 = tf.layers.dense(state_ph, 256, activation=tf.nn.relu, name='q_dense1')
					q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
					q_dense3 = tf.layers.dense(q_dense2, 256, activation=tf.nn.relu, name='q_dense3')
					q = tf.layers.dense(q_dense3, 1, name='q')
				return q

			with tf.variable_scope('main'):
				with tf.variable_scope('policy'):
					self.pi = mlp_policy(self.obs_ph)
				with tf.variable_scope('value'):
					self.q = mlp_value(self.obs_ph, self.acts_ph)
				with tf.variable_scope('value', reuse=True):
					self.q_pi = mlp_value(self.obs_ph, self.pi)

			with tf.variable_scope('target'):
				with tf.variable_scope('policy'):
					self.pi_t = mlp_policy(self.obs_next_ph)
				with tf.variable_scope('value'):
					self.q_t = mlp_value(self.obs_next_ph, self.pi_t)

		def create_operators():
			self.pi_q_loss = -tf.reduce_mean(self.q_pi)
			self.pi_l2_loss = self.args.act_l2*tf.reduce_mean(tf.square(self.pi))
			self.pi_optimizer = tf.train.AdamOptimizer(self.args.pi_lr)
			self.pi_train_op = self.pi_optimizer.minimize(self.pi_q_loss+self.pi_l2_loss, var_list=get_vars('main/policy'))

			if self.args.clip_return:
				return_value = tf.clip_by_value(self.q_t, self.args.clip_return_l, self.args.clip_return_r)
			else:
				return_value = self.q_t
			target = tf.stop_gradient(self.rews_ph+self.args.gamma*return_value)
			self.q_loss = tf.reduce_mean(tf.square(self.q-target))
			self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr)
			self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

			self.target_update_op = tf.group([
				v_t.assign(self.args.polyak*v_t + (1.0-self.args.polyak)*v)
				for v, v_t in zip(get_vars('main'), get_vars('target'))
			])

			self.saver=tf.train.Saver()
			self.init_op = tf.global_variables_initializer()
			self.target_init_op = tf.group([
				v_t.assign(v)
				for v, v_t in zip(get_vars('main'), get_vars('target'))
			])

		self.graph = tf.Graph()
		with self.graph.as_default():
			create_session()
			create_inputs()
			create_normalizer()
			create_network()
			create_operators()
		self.init_network()

	def init_network(self):
		self.sess.run(self.init_op)
		self.sess.run(self.target_init_op)

	def step(self, obs, explore=False, test_info=False):
		if (not test_info) and (self.args.buffer.steps_counter<self.args.warmup):
			return np.random.uniform(-1, 1, size=self.args.acts_dims)
		if self.args.goal_based: obs = goal_based_process(obs)

		# eps-greedy exploration
		if explore and np.random.uniform()<=self.args.eps_act:
			return np.random.uniform(-1, 1, size=self.args.acts_dims)

		feed_dict = {
			self.raw_obs_ph: [obs]
		}
		action, info = self.sess.run([self.pi, self.step_info], feed_dict)
		action = action[0]

		# uncorrelated gaussian explorarion
		if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
		action = np.clip(action, -1, 1)

		if test_info: return action, info
		return action

	def step_batch(self, obs):
		actions = self.sess.run(self.pi, {self.raw_obs_ph:obs})
		return actions

	def feed_dict(self, batch):
		return {
			self.raw_obs_ph: batch['obs'],
			self.raw_obs_next_ph: batch['obs_next'],
			self.acts_ph: batch['acts'],
			self.rews_ph: batch['rews']
		}

	def train(self, batch):
		feed_dict = self.feed_dict(batch)
		info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
		return info

	def train_pi(self, batch):
		feed_dict = self.feed_dict(batch)
		info, _ = self.sess.run([self.train_info_pi, self.pi_train_op], feed_dict)
		return info

	def train_q(self, batch):
		feed_dict = self.feed_dict(batch)
		info, _ = self.sess.run([self.train_info_q, self.q_train_op], feed_dict)
		return info

	def normalizer_update(self, batch):
		self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

	def target_update(self):
		self.sess.run(self.target_update_op)
