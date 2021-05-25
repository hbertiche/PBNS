import os
import sys
import numpy as np
import tensorflow as tf
from math import floor

from Data.data import Data
from Model.PBNS import PBNS

from util import parse_args, model_summary
from IO import writeOBJ, writePC2Frames

"""
This script will load a PBNS checkpoint
Predict results for a set of poses (at 'Data/test.npy')
Store output animation data in 'results/' folder
	Body:
	- 'results/body.obj'
	- 'results/body.pc2'
	Outfit:
	- 'results/outfit.obj'
	- 'results/ouftit.pc2'
	- 'results/rest.pc2' -> before skinning
"""

""" PARSE ARGS """
gpu_id, name, object, body, checkpoint = parse_args(train=False)
name = os.path.abspath(os.path.dirname(__file__)) + '/results/' + name + '/'
if not os.path.isdir(name):
	os.mkdir(name)
checkpoint = os.path.abspath(os.path.dirname(__file__)) + '/checkpoints/' + checkpoint

""" PARAMS """
batch_size = 32

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" MODEL """
print("Building model...")
model = PBNS(object=object, body=body, checkpoint=checkpoint)
tgts = model.gather() # model weights
model_summary(tgts)

""" DATA """
print("Reading data...")
val_poses = 'Data/test.npy'
val_data = Data(val_poses, model._shape, model._gender, batch_size=batch_size, mode='test')
val_steps = floor(val_data._n_samples / batch_size)

""" CREATE BODY AND OUTFIT .OBJ FILES """
writeOBJ(name + 'body.obj', model._body, model._body_faces)
writeOBJ(name + 'outfit.obj', model._T, model._F)

""" EVALUATION """
print("")
print("Evaluating...")
print("--------------------------")
step = 0
for poses, G, body in val_data._iterator:
	pred = model(poses, G)
	writePC2Frames(name + 'body.pc2', body.numpy())
	writePC2Frames(name + 'outfit.pc2', pred.numpy())
	writePC2Frames(name + 'rest.pc2', (model._T[None] + model.D).numpy())

	sys.stdout.write('\r\tStep: ' + str(step + 1) + '/' + str(val_steps))
	sys.stdout.flush()
	step += 1
print("")
print("")
print("DONE!")
print("")