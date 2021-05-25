import os
import sys
import numpy as np
from scipy import sparse
import tensorflow as tf
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Layers import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from IO import *
from util import *
from values import *

class Resizer:
	# Builds models
	def __init__(self, object, body, checkpoint=None):
		"""
		Args:
		- object: name of the outfit/garment OBJ file (within the same folder as this file)
		- body: name of the body MAT file (within the same folder), as described by README.md
		- checkpoint: name of the checkpoint NPY file (optional)
		file names WITHOUT extensions ('.obj', '.mat', '.npy')
		"""	
		self._object = object
		# body data
		self._read_body(body)
		# init simplified smpl (for shape only)
		self._init_smpl()
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
		if not body_mat.endswith('.mat'): body_mat = body_mat + '.mat'
		body_data = loadInfo(os.path.abspath(os.path.dirname(__file__)) + '/' + body_mat)
		self._shape = body_data['shape']
		self._gender = body_data['gender']
	
	def _init_smpl(self):
		# path
		mpath = os.path.abspath(os.path.dirname(__file__)) + '/smpl/model_[G].mat'
		mpath = mpath.replace('[G]', 'm' if self._gender else 'f')
		# load smpl
		self._smpl = loadInfo(mpath)
		
	def _read_outfit(self):
		""" Outfit data """
		root = os.path.abspath(os.path.dirname(__file__))
		self._T, F = readOBJ(root + '/' + self._object + '.obj')
		self._F = quads2tris(F) # triangulate
		self._E = faces2edges(self._F)
		self._L = laplacianMatrix(self._F)
		self._neigh_F = neigh_faces(self._F, self._E) # edges of graph representing face connectivity
		# blend shapes prior
		self._blendshapes_prior()
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
	
	def _blendshapes_prior(self, it=100):
		tree = cKDTree(self._body(self._shape))
		idx = tree.query(self._T, n_jobs=-1)[1]
		self._BS0 = self._smpl['shapedirs'][idx]
		# smooth shapedirs
		if it:
			self._BS0 = self._BS0.reshape((-1, 3 * 10))
			for i in range(it):
				self._BS0 = self._L @ self._BS0
			self._BS0 = self._BS0.reshape((-1, 3, 10))
	
	def _body(self, shape):
		return self._smpl['v_template'] + np.einsum('a,bca->bc', shape, self._smpl['shapedirs'])
			
	def _body_tf(self, shapes):
		return self._smpl['v_template'][None] + tf.einsum('ab,cdb->acd', shapes, self._smpl['shapedirs'])

	def _flat_resize(self, shape):
		return self._T[None] + np.einsum('ab,cdb->acd', (shape - self._shape[None]), self._BS0)
		
	def _compute_edges(self, T):
		return np.sqrt(np.sum((T[:,self._E[:,0]] - T[:,self._E[:,1]]) ** 2, -1))
		
	def _compute_area(self, T):
		u = T[:,self._F[:,2]] - T[:,self._F[:,0]]
		v = T[:,self._F[:,1]] - T[:,self._F[:,0]]
		return np.linalg.norm(np.cross(u,v), axis=-1).sum(-1) / 2.0
		
	# Builds model
	def _build(self):
		# Shape MLP
		self._mlp = [
			FullyConnected((12, 32), act=tf.nn.selu, name='fc0'),
			FullyConnected((32, 32), act=tf.nn.selu, name='fc1'),
			FullyConnected((32, 32), act=tf.nn.selu, name='fc2'),
			FullyConnected((32, 32), act=tf.nn.selu, name='fc3')
		]

		# Blend Shapes matrix
		shape = self._mlp[-1].w.shape[-1], self._T.shape[0], 3
		self._dBS = tf.Variable(tf.initializers.glorot_normal()(shape), name='dBS')
			
	# Returns list of model variables
	def gather(self):
		vars = [self._dBS]
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
		try:
			self._BS0 = values['BS0']
		except:
			print("Missing BS0: expect misbehavior")
		
	def save(self, checkpoint):
		# checkpoint: path to pre-trained model
		print("\tSaving checkpoint: " + checkpoint)
		# get  TF vars values
		values = {v.name: v.numpy() for v in self.gather()}
		# save BS0
		values['BS0'] = self._BS0
		# save weights
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		np.save(checkpoint, values)

	def __call__(self, X, tightness):
		# X : smpl shape (10,)
		# tightness: tightness (2,)
		""" Body """
		B = self._body_tf(X).numpy()
		""" NUMPY """
		# flat resize
		T = self._flat_resize(X.numpy())
		# edge resize
		X_e = X.numpy()
		X_e[:,0] += tightness[:,0]
		X_e[:,1] += tightness[:,1]
		X_e[:,2:] = 0
		T_e = self._flat_resize(X_e)
		# edges
		E = self._compute_edges(T_e)
		# areas
		A = self._compute_area(T_e)
		""" TENSORFLOW """
		# Pose MLP
		X = tf.concat((X, tightness), -1)
		for l in self._mlp:
			X = l(X)
		# Blend shapes
		self.D = tf.einsum('ab,bcd->acd', X, self._dBS)
		# final resize
		T = T + self.D
		return T, E, A, B