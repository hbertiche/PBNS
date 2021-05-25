import os
import sys
from getopt import getopt
import numpy as np
import tensorflow as tf
from scipy import sparse
import scipy.io as sio
from scipy.spatial import cKDTree

from values import usage_msg

def parse_args(train=True):
	gpu_id, name, object, checkpoint = [None] * 4
	opts, args = getopt(sys.argv[1:],'g:n:c:o:b:',['gpu=','name=','checkpoint=','object=','body='])
	if args: wrong_args(args)
	for opt, arg in opts:
		if opt == '-g' or opt == '--gpu':
			gpu_id = arg
		elif opt == '-n' or opt == '--name':
			name = arg
		elif opt == '-o' or opt == '--object':
			object = arg
		elif opt == '-b' or opt == '--body':
			body = arg
		elif opt == '-c' or opt == '--checkpoint':
			checkpoint = arg
		else:
			wrong_args(arg)
	assert gpu_id is not None, 'Missing GPU id'
	assert name is not None, 'Missing model name'
	assert object is not None, 'Missing outfit .OBJ file'
	assert body is not None, 'Missing body .MAT file'
	assert train or checkpoint is not None, 'Missing model checkpoint'
	return gpu_id, name, object, body, checkpoint
	
def wrong_args(args):
	print("")
	print("Unrecognized argument(s): ", args)
	print("Usage:")
	print(usage_msg)
	sys.exit(1)
	
def loadInfo(filename):
	'''
	this function should be called instead of direct sio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	'''
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	del data['__globals__']
	del data['__header__']
	del data['__version__']
	return _check_keys(data)

def _check_keys(dict):
	'''
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	'''
	for key in dict:
		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
			dict[key] = _todict(dict[key])
	return dict        

def _todict(matobj):
	'''
	A recursive function which constructs from matobjects nested dictionaries
	'''
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, sio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		elif isinstance(elem, np.ndarray) and np.any([isinstance(item, sio.matlab.mio5_params.mat_struct) for item in elem]):
			dict[strg] = [None] * len(elem)
			for i,item in enumerate(elem):
				if isinstance(item, sio.matlab.mio5_params.mat_struct):
					dict[strg][i] = _todict(item)
				else:
					dict[strg][i] = item
		else:
			dict[strg] = elem
	return dict

def rodrigues(r):
	theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
	# avoid zero divide
	theta = np.maximum(theta, np.finfo(np.float64).tiny)
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
	
def model_summary(targets):
	print("")
	_print = lambda x: print('\t' + x)
	sep = '---------------------------'
	total = 0
	_print(sep)
	_print('MODEL SUMMARY')
	_print(sep)
	for t in targets:
		_print(t.name + '\t' + str(t.shape))
		total += np.prod(t.shape)
	_print(sep)
	_print('Total params: ' + str(total))
	_print(sep)
	print("")
	
def quads2tris(F):
	F_out = []
	for f in F:
		if len(f) <= 3: F_out += [f]
		elif len(f) == 4:
			F_out += [
				[f[0], f[1], f[2]],
				[f[0], f[2], f[3]]
			]
		else:
			print('This should not happen, but might')
			print('To solve: extend this to deal with 5-gons or ensure mesh is quads/tris only')
			sys.exit()
	return np.array(F_out, np.int32)
	
def faces2edges(F):
	E = set()
	for f in F:
		N = len(f)
		for i in range(N):
			j = (i + 1) % N
			E.add(tuple(sorted([f[i], f[j]])))
	return np.array(list(E), np.int32)

def edges2graph(E):
	G = {}
	for e in E:
		if not e[0] in G: G[e[0]] = {}
		if not e[1] in G: G[e[1]] = {}
		G[e[0]][e[1]] = 1
		G[e[1]][e[0]] = 1
	return G
	
def neigh_faces(F, E=None):
	if E is None: E = faces2edges(F)
	G = {tuple(e): [] for e in E}
	for i,f in enumerate(F):
		n = len(f)
		for j in range(n):
			k = (j + 1) % n
			e = tuple(sorted([f[j], f[k]]))
			G[e] += [i]
	neighF = []
	for key in G:
		if len(G[key]) == 2:
			neighF += [G[key]]
		elif len(G[key]) > 2:
			print("Neigh F unexpected behaviour")
			continue
	return np.array(neighF, np.int32)
	
def weights_prior(T, B, weights):
	tree = cKDTree(B)
	_, idx = tree.query(T)
	return weights[idx]