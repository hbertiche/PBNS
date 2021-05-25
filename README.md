PBNS: Physically Based Neural Simulation for Unsupervised Outfit Pose Space Deformation.
<ul>
  <li><b>Unsupervised:</b> No need of gathering costly PBS data for training your models. Furthermore, PBNS qualitatively outperforms supervised learning.</li>
  <li><b>Physical consistency:</b> PBNS can generate collision-free cloth-consistent predictions. Cloth consistency ensures no texture distortion or noisy surfaces.</li>
  <li><b>Cloth properties:</b> Define per-vertex cloth properties. Neurally simulate different fabrics within the same outfit.</li>
  <li><b>Multiple layers of cloth:</b> PBNS is the only approach able to explicitly handle multiple layers of cloth. This allows modelling whole outfits, instead of single garments.</li>
  <li><b>Complements:</b> PBNS is not limited to garments. Outfits can be complemented with gloves, boots and more. This defines a common framework for outfit animation.</li>
  <li><b>Extremely efficient:</b> PBNS can achieve over 14.000 FPS (for an outfit with 24K triangles!). No previous work comes even close to this level of performance. Additionally, the memory footprint of the model is just a few MBs. Because of this, PBNS is the only solution that can be applied in real scenarios like videogames and smartphones (portable virtual try-ons). Moreover, training takes barely a few minutes, even without a GPU!</li>
  <li><b>Simple formulation:</b> PBNS formulation is the standard on 3D animation. Thus, integrating our approach into ANY rendering pipeline requires minimal engineering effort.</li>
</ul>

This repository contains the necessary code to run the model described in:<br>
https://arxiv.org/abs/2012.11310

<img src="https://sergioescalera.com/wp-content/uploads/2021/01/clothed31.png">

<img src="https://drive.google.com/uc?export=view&id=1B_rPJz3qyyf6B3py7fQ6sE5n749n8k-C">

Video:<br>
https://youtu.be/ALwhjm40zRg

<h3>Outfit resizing</h3>

PBNS formulation also allows unsupervised outfit resizing. That is, retargetting to the desired body shape and control over tightness.<br>
Just as standard PBNS, it can deal with complete outfits with multiple layers of cloth, different fabrics, complements, ...

<p float='left'>
  <img width=400px src="https://drive.google.com/uc?export=view&id=1xKkLufCBdlHaodpBpci_0SLSrv1MStFU">
  <img width=400px src="https://drive.google.com/uc?export=view&id=1V2iV38BYS8V0rl4zu72SDGrWvsyOXEjP">
</p>

<h3>Enhancing 3D Avatars</h3>

Due to the simple formulation of PBNS and no dependency from data, it can be used to easily enhance any 3D custom avatar with realistic outfits in a matter of minutes!.

<p float='left'>
  <img width=300px src="https://drive.google.com/uc?export=view&id=1JMb8Zd5BS51_hfMUGDqsDVYJqH_AY9Z7">
  <img width=300px src="https://drive.google.com/uc?export=view&id=1XdJma5ewiHvojHH-wGFISx4csHJ7So-Y">
  <img width=300px src="https://drive.google.com/uc?export=view&id=1qgfleepwAeC3EVtat8oB54DbNcl9jA9d">
</p>

<h3>Repository structure</h3>
This repository is split into two folders. One is the standard PBNS for outfit animation. The other contains the code for PBNS as a resizer.<br>
Within each folder, you will find instructions on how to use each model.
