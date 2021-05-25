import sys
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
from scipy.spatial import cKDTree
import ray
	
ray.init()	
	
def edge_loss(x, y_e, E, weights):
	x_e = tf.gather(x, E[:,0], axis=1) - tf.gather(x, E[:,1], axis=1)
	x_e = tf.sqrt(tf.reduce_sum(x_e ** 2, -1))
	d_e = (x_e - y_e)
	err = tf.reduce_mean(tf.abs(d_e))
	d_e = d_e ** 2
	d_e = tf.multiply(weights[None], d_e)
	loss = tf.reduce_sum(d_e)
	return loss, err
	
def bend_loss(x, F, neighF, weights):
	# compute face normals
	V02 = tf.gather(x, F[:,2], axis=1) - tf.gather(x, F[:,0], axis=1) # B x N x 3
	V01 = tf.gather(x, F[:,1], axis=1) - tf.gather(x, F[:,0], axis=1) # B x N x 3
	N = tf.linalg.cross(V02, V01) # B x N x 3
	N = N / tf.linalg.norm(N, axis=-1, keepdims=True) # B x N x 3
	# compare neighbouring face normals
	N0 = tf.gather(N, neighF[:,0], axis=1) # B x N x 3
	N1 = tf.gather(N, neighF[:,1], axis=1) # B x N x 3
	# cosinus distances
	cos = tf.reduce_sum(tf.multiply(N0, N1), -1)
	err = tf.reduce_mean(1 - cos)
	dN = (N0 - N1) ** 2
	dN = tf.multiply(weights[None,:,None], dN)
	loss = tf.reduce_sum(dN)
	return loss, err

""" COLLISION FUNCTIONS START """
@ray.remote
def _nn(foo):
	return cKDTree(foo[1]).query(foo[0].numpy())[1]
	
def collision_loss(V, F, B, B_F, layers, thr=.004, stop_gradient=False):
	V_vn = tfg.geometry.representation.mesh.normals.vertex_normals(V, tf.tile(F[None], [V.shape[0], 1, 1]))
	B_vn = tfg.geometry.representation.mesh.normals.vertex_normals(B, tf.tile(B_F[None], [B.shape[0], 1, 1]))
	loss = 0
	vcount = np.array([0] * len(layers), np.float32)
	for i,l in enumerate(layers):
		# get layer verts
		v = tf.gather(V, l, axis=1)
		# compute correspondences
		idx = ray.get([_nn.remote((v[n],B[n])) for n in range(v.shape[0])])
		N, M = v.shape[:2]
		_idx = tf.tile(np.array(list(range(N)), np.int32)[:,None], [1, M])
		idx = tf.stack((_idx, idx), axis=-1)
		# compute loss
		D = v - tf.gather_nd(B, idx)
		b_vn = tf.gather_nd(B_vn, idx)
		dot = tf.einsum('abc,abc->ab', D, b_vn)
		loss += tf.reduce_sum(tf.minimum(dot - thr, 0) ** 2)
		# vcount
		vmask = tf.cast(tf.math.less(dot, 0), tf.float32)
		vcount[i] = tf.reduce_sum(vmask).numpy() / (N * M)
		# add layer to "body"
		if i + 1 < len(layers):
			if stop_gradient:
				v = tf.stop_gradient(v)
				v_vn = tf.stop_gradient(tf.gather(V_vn, l, axis=1))
			else:
				v_vn = tf.gather(V_vn, l, axis=1)
			B = tf.concat((B, v), axis=1)
			B_vn = tf.concat((B_vn, v_vn), axis=1)
	return loss, vcount
""" COLLISION FUNCTIONS END """

def gravity_loss(x, surface, m=.15, g=9.81):
	vertex_mass = surface * m / x.shape[1]
	U = vertex_mass[:,None] * g * x[:,:,2]
	return tf.reduce_sum(U), tf.reduce_mean(tf.reduce_sum(U, -1))