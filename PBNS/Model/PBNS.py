import os
import sys
import numpy as np
from scipy import sparse
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Layers import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from IO import *
from util import *
from values import *

class PBNS:
	# Builds models
	def __init__(self, object, body, checkpoint=None, blendweights=False):
		"""
		Args:
		- object: name of the outfit/garment OBJ file (within the same folder as this file)
		- body: name of the body MAT file (within the same folder), as described by README.md
		- checkpoint: name of the checkpoint NPY file (optional)
		- blendweights: True or False. Optimize blendweights as well?
		file names WITHOUT extensions ('.obj', '.mat', '.npy')
		"""	
		self._object = object
		# optimize blendweights?
		self._blendweights = blendweights
		# body data
		self._read_body(body)
		# outfit data
		self._read_outfit()
		# build
		self._build()
		# load pre-trained
		# the code does not check if checkpoint, object and body are consistent
		if checkpoint is not None:
			print("Loading pre-trained model: " + checkpoint)
			self.load(checkpoint)

	
	def _read_body(self, body_mat):
		# 'body_mat' should be the name of a MAT file with:
		# 'body' in rest pose (6890 x 3)
		# 'faces' triangulated (13776 x 3)
		# 'blendweights' as in SMPL (6890 x 24)
		# 'shape' as in SMPL (10)
		# 'gender' 0 = female, 1 = male
		if not body_mat.endswith('.mat'): body_mat = body_mat + '.mat'
		body_data = loadInfo(os.path.abspath(os.path.dirname(__file__)) + '/' + body_mat)
		self._body = body_data['body']
		self._body_faces = np.int32(body_data['faces'])
		self._body_weights = body_data['blendweights']
		self._shape = body_data['shape']
		self._gender = body_data['gender']
		
	def _read_outfit(self):
		""" Outfit data """
		# reads outfit .OBJ file
		# pre-computes data-structures and values
		# edge indices as |E| x 2 array
		# edge lengths as |E|
		# faces connectivity as the edges of a face connectivity graph (|EF| x 2)
		# initial outfit blend weights by proximity to SMPL body
		root = os.path.abspath(os.path.dirname(__file__))
		self._T, F = readOBJ(root + '/' + self._object + '.obj')
		self._F = quads2tris(F) # triangulate
		self._E = faces2edges(self._F)
		self._neigh_F = neigh_faces(self._F, self._E) # edges of graph representing face connectivity
		# edge lengths on rest outfit mesh
		self._precompute_edges()
		# cloth area (to compute mass later)
		self._precompute_area()
		# blend weights initial value
		self._W = weights_prior(self._T, self._body, self._body_weights)
		self._W /= np.sum(self._W, axis=-1, keepdims=True)
		self._W = np.float32(self._W)
		""" Outfit config """
		# - 'layers': list of lists. Each list represents a layer. Each sub-list contains the indices of the vertices belonging to that layer.
		# - 'edge': per-vertex weights for edge loss
		# - 'bend': per-vertex weights for bending loss
		# - 'pin': pinned vertex indices
		N = self._T.shape[0] # n. verts
		self._config = {}		
		config_path = root + '/' + self._object + '_config.mat'
		if os.path.isfile(config_path):
			self._config = loadInfo(config_path)
		else:
			print("Outfit config file not found. Using default config.")
		# DEFAULT MISSING CONFIG FIELDS
		# layers
		if 'layers' not in self._config:
			self._config['layers'] = [list(range(N))]
		# edge
		if 'edge' not in self._config:
			self._config['edge'] = np.ones((N,), np.float32)
		# bend
		if 'bend' not in self._config:
			self._config['bend'] = np.ones((N,), np.float32)
	
		# convert per-vertex 'edge' weights to per-edge weights
		_edge_weights = np.zeros((len(self._E),), np.float32)
		for i,e in enumerate(self._E):
			_edge_weights[i] = self._config['edge'][e].mean()
		self._config['edge'] = _edge_weights
		# convert per-vertex 'bend' weights to per-hinge weights (hinge = adjacent faces)
		_bend_weights = np.zeros((len(self._neigh_F),), np.float32)
		for i,n_f in enumerate(self._neigh_F):
			# get common verts
			v = list(set(self._F[n_f[0]]).intersection(set(self._F[n_f[1]])))
			_bend_weights[i] = self._config['bend'][v].mean()
		self._config['bend'] = _bend_weights
		
	# computes rest outfit edge lengths
	def _precompute_edges(self):
		T, E = self._T, self._E
		e = tf.gather(T, E[:,0], axis=0) - tf.gather(T, E[:,1], axis=0)
		self._edges = tf.sqrt(tf.reduce_sum(e ** 2, -1))
	
	# computes rest outfit total area
	def _precompute_area(self):
		T, F = self._T, self._F
		u = tf.gather(T, F[:,2], axis=0) - tf.gather(T, F[:,0], axis=0)
		v = tf.gather(T, F[:,1], axis=0) - tf.gather(T, F[:,0], axis=0)
		areas = tf.norm(tf.linalg.cross(u, v), axis=-1)
		self._total_area = tf.reduce_sum(areas) / 2.0
		print("Total cloth area: ", self._total_area.numpy())
		
	# Builds model
	def _build(self):
		# Blendweights
		self._W = tf.Variable(self._W, name='blendweights', trainable=self._blendweights)
		
		# Pose MLP
		self._mlp = [
			FullyConnected((72, 32), act=tf.nn.relu, name='fc0'),
			FullyConnected((32, 32), act=tf.nn.relu, name='fc1'),
			FullyConnected((32, 32), act=tf.nn.relu, name='fc2'),
			FullyConnected((32, 32), act=tf.nn.relu, name='fc3')
		]

		# PSD
		shape = self._mlp[-1].w.shape[-1], self._T.shape[0], 3
		self._psd = tf.Variable(tf.initializers.glorot_normal()(shape), name='psd')
			
	# Returns list of model variables
	def gather(self):
		vars = [self._psd, self._W]
		for l in self._mlp:
			vars += l.gather()
		return vars
	
	# loads pre-trained model
	def load(self, checkpoint):
		# checkpoint: path to pre-trained model
		# list vars
		vars = self.gather()
		# load vars values
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		values = np.load(checkpoint, allow_pickle=True)[()]
		# assign
		for v in vars:
			try: 
				v.assign(values[v.name])
			except: print("Mismatch between model and checkpoint: " + v.name)
		
	def save(self, checkpoint):
		# checkpoint: path to pre-trained model
		print("\tSaving checkpoint: " + checkpoint)
		# get vars values
		values = {v.name: v.numpy() for v in self.gather()}
		# save weights
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		np.save(checkpoint, values)

	# Computes the skinning for each outfit/pose
	def _skinning(self, T, G):
		W = self._W
		W = tf.nn.elu(1e2 * self._W) # allows back-propagating gradients to zero-valued blend weights
		W = tf.divide(W, tf.reduce_sum(W, -1, keepdims=True))
		G = tf.einsum('ab,cbde->cade', W, G)
		return tf.einsum('abcd,abd->abc', G, self._with_ones(T))[:,:,:3]
	
	def _with_ones(self, X):
		return tf.concat((X, tf.ones((*X.shape[:2], 1), tf.float32)), axis=-1)

	def __call__(self, X, G):
		# X : poses as (B x 72)
		# G : joint transformation matrices as (B x 24 x 4 x 4)
		# Pose MLP
		for l in self._mlp:
			X = l(X)
		# PSD
		self.D = tf.einsum('ab,bcd->acd', X, self._psd)
		# Compute skinning
		return self._skinning(self._T[None] + self.D, G)
