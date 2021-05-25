import os
import sys
from random import shuffle, choice
from math import floor
from scipy import sparse

from time import time
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from util import *
from IO import *
from values import *

class Data:	
	def __init__(self, gender, shape_range=3, tightness_range=1, epoch_steps=16000, batch_size=10, mode='train'):
		"""
		Args:
		- gender: 0 = female, 1 = male
		- shape_range: range of values for SMPL shape parameters (-shape_range, shape_range)
		- tightness_range: as shape_range for tightness
		- epoch_steps: number of steps per epoch
		- batch_size: batch size
		- shuffle: shuffle
		"""
		self._shape_range = shape_range
		self._tightness_range = tightness_range
		# TF Dataset
		ds = tf.data.Dataset.from_tensor_slices(np.arange(epoch_steps).reshape((-1,1)))
		ds = ds.map(self.tf_map, num_parallel_calls=batch_size)
		ds = ds.batch(batch_size=batch_size)
		if mode == 'train':
			ds = ds.shuffle(batch_size)
		self._iterator = ds
		self._n_samples = epoch_steps
		
	def _next(self, i):
		# random shape
		shape = self._shape_range * np.random.uniform(-1, 1, size=(10,))
		tightness = self._tightness_range * np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
		# shape
		return shape, tightness
		
	def tf_map(self, i):
		return tf.py_function(func=self._next, inp=[i], Tout=[tf.float32, tf.float32])