<h3>Train script usage</h3>

<b>GPU</b><br>
-g, --gpu<br>
ID of the GPU to use.

<b>Name</b><br>
-n, --name<br>
Name under which model checkpoints will be stored. Stored in 'checkpoints/' folder.<br>

<b>Object</b><br>
-o, --object<br>
Name of the OBJ file (without '.obj') with the outfit in rest pose (rest pose defined in 'values.py').<br>
It expects the OBJ file to be placed at 'Model/' folder.<br>
Optionally, place a MAT file like '[objectname]_config.mat' with a the following fields:<br>
<ul>
	<li>'pin': list of vertex indices. Mark these vertices as <i>pinned down</i>. This will limit their movement.</li>
	<li>'edge': per-vertex weights for edge loss term.</li>
	<li>'bend': per-vertex weights for bending loss term.</li>
	<li>'layers': list of lists. Defines the order for multiple layers of cloth. From inner to outer. Sub-lists must contain vertex indices of each layer.</li>
</ul>
All fields are optional. Default config will be used for missing fields.<br>
If this file exists, it will use it. Otherwise it will use a default config.<br>
A sample outfit is provided with its corresponding config data. At 'Model/Outfit.obj' and 'Model/Outfit_config.mat'.<br>

<b>Body</b><br>
-b, --body<br>
Name of the MAT file (without '.mat') with SMPL body data like:
<ul>
  <li>'shape': shape parameters as 10-dimensional array</li>
  <li>'gender': gender (0 = female, 1 = male)</li>
</ul>
Note that the outfit to simulate should be aligned with the body in rest pose, as represented by these data.<br>
A sample body is provided, aligned with the provided outfit. At 'Model/Body.mat' (rest pose defined in 'values.py'). <br>

<b>Checkpoint (Optional)</b><br>
-c, --checkpoint<br>
Name of '.npy' file (without '.npy') with pre-trained weights for the network.
It expects the checkpoints to be placed at 'checkpoints/' folder.<br>

Example of usage:<br>
python train.py -g 0 -b Body -o Outfit -n MyModel -c MyCheckpoint<br>

To change other parameters (hyperparameters or simulation properties), you will find them at the beginning of the 'train.py' script.


<h3>Evaluation script usage</h3>

<b>GPU</b><br>
-g, --gpu<br>
ID of the GPU to use.

<b>Name</b><br>
-n, --name<br>
Name under which model results will be stored. Stored in 'results/name/' folder. You might need to manually create 'results/' folder first.<br>
Results are stored as:
<ul>
  <li>Outfit OBJ file (original shape)</li>
  <li>Outfit PC2 file (animation data)</li>
  <li>Body OBJ file (original shape)</li>
  <li>Body PC2 file (animation data)</li>
</ul>

<b>Object</b><br>
-o, --object<br>
Same as before. Needs to correspond to the object used for training the checkpoint to evaluate.<br>

<b>Body</b><br>
-b, --body<br>
Same as before. Needs to correspond to the body used for training the checkpoint to evaluate.<br>

<b>Checkpoint</b><br>
-c, --checkpoint<br>
Name of '.npy' file (without '.npy') with pre-trained weights for the network.
It expects the checkpoints to be placed at 'checkpoints/' folder.<br>
Note that this time this is not an optional argument.<br>

<h3>Shape data</h3>
Shape and tightness data can be randomly generated.<br>
Exhaustive training on the whole input domain takes significantly longer than standard PBNS (few hours).

<h3>SMPL</h3>
Simplified versions of SMPL for reshaping only are already provided within 'Model/smpl/'.