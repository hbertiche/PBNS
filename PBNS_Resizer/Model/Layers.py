import tensorflow as tf

class FullyConnected:
	def __init__(self, shape, act=None, name='fc'):
		self.w = tf.Variable(tf.initializers.glorot_normal()(shape), name=name+'_w')
		self.b = tf.Variable(tf.zeros(shape[-1], dtype=tf.float32), name=name+'_b')
		# self.b = tf.Variable(tf.initializers.glorot_normal()((shape[-1],)), name=name+'_b')
		self.act = act or (lambda x: x)

	def gather(self):
		return [self.w, self.b]

	def __call__(self, X):
		X = tf.einsum('ab,bc->ac', X, self.w) + self.b
		X = self.act(X)
		return X