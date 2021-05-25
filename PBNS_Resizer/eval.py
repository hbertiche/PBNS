import os
import sys
import numpy as np
import tensorflow as tf
from math import floor

from Data.data import Data
from Model.Resizer import Resizer
from Losses import *

from util import *
from IO import writeOBJ, writePC2Frames

def linspace(S, T, steps):
	assert len(S) == len(T), 'Wrong dimensions: ' + str(len(S)) + ' / ' + str(len(T))
	X = np.concatenate((S,T), axis=-1)
	X = np.concatenate([np.linspace(X[i], X[i + 1], steps) for i in range(X.shape[0] - 1)], axis=0)
	return X[:,:10], X[:,10:]

"""
This script will load a PBNS Resizer checkpoint
Generate random shape/tightness values and interpolate
Store output animation data in 'results/' folder
	Body:
	- 'results/body.obj'
	- 'results/body.pc2'
	Outfit:
	- 'results/outfit.obj'
	- 'results/ouftit.pc2'
"""
	
""" PARSE ARGS """
gpu_id, name, object, body, checkpoint = parse_args()
name = os.path.abspath(os.path.dirname(__file__)) + '/results/' + name + '/'
if not os.path.isdir(name):
	os.mkdir(name)
checkpoint = os.path.abspath(os.path.dirname(__file__)) + '/checkpoints/' + checkpoint
	
""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" RANGES """
shape_range = 5
tightness_range = 2

""" MODEL """
print("Building model...")
model = Resizer(object=object, body=body, checkpoint=checkpoint)
tgts = model.gather() # model weights
tgts = [v for v in tgts if v.trainable]
model_summary(tgts)

""" DATA """
# Generates 'n_points' random shape/tightness values and smoothly interpolates 'step' intermediate steps
print("Generating data...")
n_points = 30
shapes = []
tightness = []
for i in range(n_points):
	shapes += [shape_range * np.random.uniform(-1, 1, size=(10,))]
	tightness += [tightness_range * np.random.uniform(-1, 1, size=(2,))]
steps = 10
shapes, tightness = linspace(shapes, tightness, steps)

""" CREATE BODY AND OUTFIT .OBJ FILES """
writeOBJ(name + 'body.obj', model._smpl['v_template'], model._smpl['faces'])
writeOBJ(name + 'outfit.obj', model._T, model._F)

print("")
print("Evaluating...")
print("--------------------------")
step = 0
steps = shapes.shape[0]
for s,t in zip(shapes, tightness):
	s = tf.constant(s[None], tf.float32)
	t = tf.constant(t[None], tf.float32)
	pred, E, A, B = model(s, t)
	
	writePC2Frames(name + 'body.pc2', B)
	writePC2Frames(name + 'outfit.pc2', pred.numpy())
	
	sys.stdout.write('\r\tStep: ' + str(step + 1) + '/' + str(steps))
	sys.stdout.flush()
	step += 1
print("")
print("")
print("DONE!")
print("")