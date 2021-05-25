import sys
import numpy as np
import pickle

class SMPLModel():
	def __init__(self, model_path, rest_pose=None):
		"""
		SMPL model.

		Parameter:
		---------
		model_path: Path to the SMPL model parameters, pre-processed by
		`preprocess.py`.

		"""
		self._rest_pose = rest_pose.reshape((-1, 1, 3))
		with open(model_path, 'rb') as f:
			if sys.version_info[0] == 2: 
				params = pickle.load(f) # Python 2.x
			elif sys.version_info[0] == 3: 
				params = pickle.load(f, encoding='latin1') # Python 3.x
			self.J_regressor = params['J_regressor']
			self.weights = params['weights']
			self.posedirs = params['posedirs']
			self.v_template = params['v_template']
			self.shapedirs = params['shapedirs']
			self.faces = np.int32(params['f'])
			self.kintree_table = params['kintree_table']

		id_to_col = {
			self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
		}
		self.parent = {
			i: id_to_col[self.kintree_table[0, i]]
			for i in range(1, self.kintree_table.shape[1])
		}

	def set_params(self, pose=None, beta=None, trans=None, with_body=False):
		"""
		Set pose, shape, and/or translation parameters of SMPL model. Verices of the
		model will be updated and returned.

		Parameters:
		---------
		pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
		relative to parent joint. For root joint it's global orientation.
		Represented in a axis-angle format.

		beta: Parameter for model shape. A vector of shape [10]. Coefficients for
		PCA component. Only 10 components were released by MPI.

		trans: Global translation of shape [3].

		Return:
		------
		Updated vertices.

		"""
		# posed body
		G, B = self.update(pose, beta, with_body)
		# rest pose body
		G_rest, _ = self.update(self._rest_pose, beta, with_body=False)
		# from rest to pose
		for i in range(G.shape[0]):
			G[i] = G[i] @ np.linalg.inv(G_rest[i])
		return G, B

	def update(self, pose, beta, with_body):
		"""
		Called automatically when parameters are updated.

		"""
		# how beta affect body shape
		v_shaped = self.shapedirs.dot(beta) + self.v_template
		# joints location
		J = self.J_regressor.dot(v_shaped)
		# align root joint with origin
		v_shaped -= J[:1]
		J -= J[:1]
		pose_cube = pose.reshape((-1, 1, 3))
		# rotation matrix for each joint
		R = self.rodrigues(pose_cube)
		# world transformation of each joint
		G = np.empty((self.kintree_table.shape[1], 4, 4))
		G[0] = self.with_zeros(np.hstack((R[0], J[0, :].reshape([3, 1]))))
		for i in range(1, self.kintree_table.shape[1]):
			G[i] = G[self.parent[i]].dot(
				self.with_zeros(
					np.hstack(
						[R[i],((J[i, :]-J[self.parent[i],:]).reshape([3,1]))]
					)
				)
			)
		G = G - self.pack(
			np.matmul(
				G,
				np.hstack([J, np.zeros([24, 1])]).reshape([24, 4, 1])
				)
			)
		v = None
		if with_body:
			I_cube = np.broadcast_to(
				np.expand_dims(np.eye(3), axis=0),
				(R.shape[0]-1, 3, 3)
			)
			lrotmin = (R[1:] - I_cube).ravel()
			# how pose affect body shape in zero pose
			v_posed = v_shaped + self.posedirs.dot(lrotmin)	
			# transformation of each vertex
			T = np.tensordot(self.weights, G, axes=[[1], [0]])
			rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
			v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
			
		return G, v

	def rodrigues(self, r):
		"""
		Rodrigues' rotation formula that turns axis-angle vector into rotation
		matrix in a batch-ed manner.

		Parameter:
		----------
		r: Axis-angle rotation vector of shape [batch_size, 1, 3].

		Return:
		-------
		Rotation matrix of shape [batch_size, 3, 3].

		"""
		theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
		# avoid zero divide
		theta = np.maximum(theta, np.finfo(np.float64).eps)
		r_hat = r / theta
		cos = np.cos(theta)
		z_stick = np.zeros(theta.shape[0])
		m = np.dstack([
			z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
			r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
			-r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
		).reshape([-1, 3, 3])
		i_cube = np.broadcast_to(
			np.expand_dims(np.eye(3), axis=0),
			[theta.shape[0], 3, 3]
		)
		A = np.transpose(r_hat, axes=[0, 2, 1])
		B = r_hat
		dot = np.matmul(A, B)
		R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
		return R

	def with_zeros(self, x):
		"""
		Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

		Parameter:
		---------
		x: Matrix to be appended.

		Return:
		------
		Matrix after appending of shape [4,4]

		"""
		return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

	def pack(self, x):
		"""
		Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
		manner.

		Parameter:
		----------
		x: Matrices to be appended of shape [batch_size, 4, 1]

		Return:
		------
		Matrix of shape [batch_size, 4, 4] after appending.

		"""
		return np.dstack((np.zeros((x.shape[0], 4, 3)), x))