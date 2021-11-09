import os
import sys
from random import shuffle, choice
from math import floor
from scipy import sparse

from time import time
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smpl.smpl_np import SMPLModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from util import *
from IO import *
from values import *

class Data:	
	def __init__(self, poses, shape, gender, batch_size=10, mode='train'):
		"""
		Args:
		- poses: path to .npy file with poses
		- shape: SMPL shape parameters for the subject
		- gender: 0 = female, 1 = male
		- batch_size: batch size
		- shuffle: shuffle
		"""
		# Read sample list
		self._poses = np.load(poses)	
		if self._poses.dtype == np.float64: self._poses = np.float32(self._poses)
		self._n_samples = self._poses.shape[0]
		# smpl
		smpl_path = os.path.dirname(os.path.abspath(__file__)) + '/smpl/'
		smpl_path += 'model_[G].pkl'.replace('[G]', 'm' if gender else 'f')
		self.SMPL = SMPLModel(smpl_path, rest_pose)
		self._shape = shape
		# TF Dataset
		ds = tf.data.Dataset.from_tensor_slices(self._poses)
		if mode == 'train': ds = ds.shuffle(self._n_samples)
		ds = ds.map(self.tf_map, num_parallel_calls=batch_size)
		ds = ds.batch(batch_size=batch_size)
		self._iterator = ds

	def _next(self, pose):
		# compute body
		# while computing SMPL should be part of PBNS, 
		# if it is in Data, it can be efficiently parallelized without overloading GPU
		G, B = self.SMPL.set_params(pose=pose.numpy(), beta=self._shape, with_body=True)
		return pose, G, B
		
	def tf_map(self, pose):
		return tf.py_function(func=self._next, inp=[pose], Tout=[tf.float32, tf.float32, tf.float32])
