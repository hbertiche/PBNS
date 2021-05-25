import os
import sys
import numpy as np
import tensorflow as tf

from time import time
from datetime import timedelta
from math import floor

from Data.data import Data
from Model.Resizer import Resizer
from Losses import *
# use 'import' below for parallelized collision loss (specially for bigger batches)
# from LossesRay import *

from util import *
from IO import writePC2Frames

""" PARSE ARGS """
gpu_id, name, object, body, checkpoint = parse_args()
if checkpoint is not None:
	checkpoint = os.path.abspath(os.path.dirname(__file__)) + '/checkpoints/' + checkpoint
	
""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" Log """
stdout_steps = 100 # update stdout every N steps
if name == 'test': stdout_steps = 1

""" TRAIN PARAMS """
batch_size = 16
num_epochs = 1000
epoch_steps = batch_size * 1000

""" RESIZING PARAMETERS """
shape_range = 5
tightness_range = 2 # if unstable, train with lower value and finetune later

""" SIMULATION PARAMETERS """
edge_young = 5
bend_weight = 1 * 1e-5
collision_weight = 25
collision_dist = .004
mass = .3 # fabric mass as surface density (kg / m2)

""" MODEL """
print("Building model...")
model = Resizer(object=object, body=body, checkpoint=checkpoint)
tgts = model.gather() # model weights
tgts = [v for v in tgts if v.trainable]
model_summary(tgts)
optimizer = tf.optimizers.SGD(lr=1e-5, momentum=.9)

""" DATA """
print("Generating data pipeline...")
tr_data = Data(model._gender, shape_range=shape_range, tightness_range=tightness_range, batch_size=batch_size, epoch_steps=epoch_steps)

tr_steps = floor(tr_data._n_samples / batch_size)
for epoch in range(num_epochs):
	print("")
	print("Epoch " + str(epoch + 1))
	print("--------------------------")
	""" TRAIN """
	print("Training...")
	total_time = 0
	step = 0
	metrics = [0] * 4 # Edge, Bend, Gravity, Collisions
	start = time()	
	for shapes, tightness in tr_data._iterator:
		""" Train step """
		with tf.GradientTape() as tape:
			pred, E, A, body = model(shapes, tightness)
			# Losses & Metrics			
			# cloth
			L_edge, E_edge = edge_loss(pred, E, model._E, weights=model._config['edge'])
			L_bend, E_bend = bend_loss(pred, model._F, model._neigh_F, weights=model._config['bend'])
			# collision
			L_collision, E_collision = collision_loss(pred, model._F, body, model._smpl['faces'], model._config['layers'], thr=collision_dist)
			# gravity
			L_gravity, E_gravity = gravity_loss(pred, surface=A, m=mass)
			# pin
			if 'pin' in model._config:
				L_pin = tf.reduce_sum(tf.gather(model.D, model._config['pin'], axis=1) ** 2)
			else:
				L_pin = tf.constant(0.0)
			loss =  edge_young * L_edge + \
					bend_weight * L_bend + \
					collision_weight * L_collision + \
					L_gravity + \
					1e1 * L_pin
		""" Backprop """
		grads = tape.gradient(loss, tgts)		
		optimizer.apply_gradients(zip(grads, tgts))
		""" Progress """
		metrics[0] += E_edge.numpy()
		metrics[1] += E_bend.numpy()
		metrics[2] += E_gravity.numpy()
		metrics[3] += E_collision
		total_time = time() - start
		
		ETA = (tr_steps - step - 1) * (total_time / (1+step))
		if (step + 1) % stdout_steps == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(tr_steps) + ' ... '
					+ 'E: {:.2f}'.format(1000 * metrics[0] / (1+step)) # in millimeters
					+ ' - '
					+ 'B: {:.3f}'.format(metrics[1] / (1+step))
					+ ' - '	
					+ 'G: {:.4f}'.format(1000 * metrics[2] / (1+step)) # in mJ
					+ ' - '
					+ 'C: [' + ', '.join(['{:.4f}'.format(m / (1+step)) for m in metrics[3]]) + ']'
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
		step += 1
	""" Epoch results """
	metrics = [m / step for m in metrics]
	print("")
	print("Total edge: {:.5f}".format(1000 * metrics[0]))
	print("Total bending: {:.5f}".format(metrics[1]))
	print("Total gravity: {:.2f}".format(metrics[2]))
	print("Total collision: [" + ', '.join(['{:.4f}'.format(m) for m in metrics[3]]) + ']')
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" Save checkpoint """
	model.save('checkpoints/' + name)