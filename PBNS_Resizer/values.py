import numpy as np

rest_pose = np.zeros((24,3))
rest_pose[0, 0] = np.pi / 2
rest_pose[1, 2] = .15
rest_pose[2, 2] = -.15

# script usage
usage_msg = "\t-g, --gpu:			id of the GPU to use,\n" + \
			"\t-n, --name:			name for the model (used to save checkpoints in train, and results in test),\n" + \
			"\t-o, --object:		.OBJ file for the garment/outfit in rest pose (script assumes metadata shares name with OBJ)\n" + \
			"\t-b, --body:			.MAT file with associated body information\n" + \
			"\t-c, --checkpoint:	name for pre-trained model (optional for train),\n"