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
  <li>'body': body vertices as 6890 x 3 (in rest pose as defined in 'values.py')</li>
  <li>'faces': triangulated body faces as 13776 x 3</li>
  <li>'blendweights': body blend weights as 6890 x 24</li>
  <li>'shape': shape parameters as 10-dimensional array</li>
  <li>'gender': gender (0 = female, 1 = male)</li>
</ul>
Note that the outfit to simulate should be aligned with the body in rest pose, as represented by these data.<br>
A sample body is provided, aligned with the provided outfit. At 'Model/Body.mat'.<br>

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
  <li>Outfit OBJ file (rest pose)</li>
  <li>Outfit PC2 file (animation data)</li>
  <li>Rest Outfit PC2 file (animation data; before skinning)</li>
  <li>Body OBJ file (rest pose)</li>
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

<h3>Pose data</h3>
To run the neural simulator it is necessary a valid source of pose data (N x 72 for SMPL).<br>
Make sure pose data is properly balanced. Instructions in the main paper, sec. 4.3.<br>
We cannot provide this data ourselves because of license issues.<br>
Details on data format are located in 'Data/README.md'.

<h3>SMPL</h3>
The script will expect SMPL models as:
<ul>
	<li>'Data/smpl/model_f.pkl'</li>
	<li>'Data/smpl/model_m.pkl'</li>
</ul>