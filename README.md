### PBNS: Physically Based Neural Simulation for Unsupervised Outfit Pose Space Deformation.

<a href="hbertiche.github.io/PBNS">Project Page</a> | <a href="https://dl.acm.org/doi/10.1145/3478513.3480479">Paper</a> | <a href="https://arxiv.org/abs/2012.11310">arXiv</a> | <a href="https://youtu.be/ALwhjm40zRg">Video</a>

<ul>
  <li><b>Unsupervised:</b> No need of gathering costly PBS data for training your models. Furthermore, PBNS qualitatively outperforms supervised learning.</li>
  <li><b>Physical consistency:</b> PBNS can generate collision-free cloth-consistent predictions. Cloth consistency ensures no texture distortion or noisy surfaces.</li>
  <li><b>Cloth properties:</b> Define per-vertex cloth properties. Neurally simulate different fabrics within the same outfit.</li>
  <li><b>Multiple layers of cloth:</b> PBNS is the only approach able to explicitly handle multiple layers of cloth. This allows modelling whole outfits, instead of single garments.</li>
  <li><b>Complements:</b> PBNS is not limited to garments. Outfits can be complemented with gloves, boots and more. This defines a common framework for outfit animation.</li>
  <li><b>Extremely efficient:</b> PBNS can achieve over 14.000 FPS (for an outfit with 24K triangles!). No previous work comes even close to this level of performance. Additionally, the memory footprint of the model is just a few MBs. Because of this, PBNS is the only solution that can be applied in real scenarios like videogames and smartphones (portable virtual try-ons). Moreover, training takes barely a few minutes, even without a GPU!</li>
  <li><b>Simple formulation:</b> PBNS formulation is the standard on 3D animation. Thus, integrating our approach into ANY rendering pipeline requires minimal engineering effort.</li>
</ul>

### Abstract

>
>
>We present a methodology to automatically obtain Pose Space Deformation (PSD) basis for rigged garments through deep learning. Classical approaches rely on Physically Based Simulations (PBS) to animate clothes. These are general solutions that, given a sufficiently fine-grained discretization of space and time, can achieve highly realistic results. However, they are computationally expensive and any scene modification prompts the need of re-simulation. Linear Blend Skinning (LBS) with PSD offers a lightweight alternative to PBS, though, it needs huge volumes of data to learn proper PSD. We propose using deep learning, formulated as an implicit PBS, to un-supervisedly learn realistic cloth Pose Space Deformations in a constrained scenario: dressed humans. Furthermore, we show it is possible to train these models in an amount of time comparable to a PBS of a few sequences. To the best of our knowledge, we are the first to propose a neural simulator for cloth. While deep-based approaches in the domain are becoming a trend, these are data-hungry models. Moreover, authors often propose complex formulations to better learn wrinkles from PBS data. Supervised learning leads to physically inconsistent predictions that require collision solving to be used. Also, dependency on PBS data limits the scalability of these solutions, while their formulation hinders its applicability and compatibility. By proposing an unsupervised methodology to learn PSD for LBS models (3D animation standard), we overcome both of these drawbacks. Results obtained show cloth-consistency in the animated garments and meaningful pose-dependant folds and wrinkles. Our solution is extremely efficient, handles multiple layers of cloth, allows unsupervised outfit resizing and can be easily applied to any custom 3D avatar.

<a href="mailto:hugo_bertiche@hotmail.com">Hugo Bertiche</a>, <a href="mailto:mmadadi@cvc.uab.cat">Meysam Madadi</a> and <a href="https://sergioescalera.com/">Sergio Escalera</a>

<img src="https://sergioescalera.com/wp-content/uploads/2021/01/clothed31.png">

<img src="/gifs/seqs.gif">

## Outfit resizing

PBNS formulation also allows unsupervised outfit resizing. That is, retargetting to the desired body shape and control over tightness.<br>
Just as standard PBNS, it can deal with complete outfits with multiple layers of cloth, different fabrics, complements, ...

<p float='left'>
  <img width=400px src="/gifs/resizer0.gif">
  <img width=400px src="/gifs/resizer1.gif">
</p>

## Enhancing 3D Avatars

Due to the simple formulation of PBNS and no dependency from data, it can be used to easily enhance any 3D custom avatar with realistic outfits in a matter of minutes!

<p float='left'>
  <img width=300px src="/gifs/avatar1.gif">
  <img width=300px src="/gifs/avatar2.gif">
  <img width=300px src="/gifs/avatar0.gif">
</p>

## Repository structure

This repository is split into two folders. One is the standard PBNS for outfit animation. The other contains the code for PBNS as a resizer.<br>
Within each folder, you will find instructions on how to use each model.


## Citation
```
@article{10.1145/3478513.3480479,
        author = {Bertiche, Hugo and Madadi, Meysam and Escalera, Sergio},
        title = {PBNS: Physically Based Neural Simulation for Unsupervised Garment Pose Space Deformation},
        year = {2021},
        issue_date = {December 2021},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {40},
        number = {6},
        issn = {0730-0301},
        url = {https://doi.org/10.1145/3478513.3480479},
        doi = {10.1145/3478513.3480479},
        abstract = {We present a methodology to automatically obtain Pose Space Deformation (PSD) basis for rigged garments through deep learning. Classical approaches rely on Physically Based Simulations (PBS) to animate clothes. These are general solutions that, given a sufficiently fine-grained discretization of space and time, can achieve highly realistic results. However, they are computationally expensive and any scene modification prompts the need of re-simulation. Linear Blend Skinning (LBS) with PSD offers a lightweight alternative to PBS, though, it needs huge volumes of data to learn proper PSD. We propose using deep learning, formulated as an implicit PBS, to un-supervisedly learn realistic cloth Pose Space Deformations in a constrained scenario: dressed humans. Furthermore, we show it is possible to train these models in an amount of time comparable to a PBS of a few sequences. To the best of our knowledge, we are the first to propose a neural simulator for cloth. While deep-based approaches in the domain are becoming a trend, these are data-hungry models. Moreover, authors often propose complex formulations to better learn wrinkles from PBS data. Supervised learning leads to physically inconsistent predictions that require collision solving to be used. Also, dependency on PBS data limits the scalability of these solutions, while their formulation hinders its applicability and compatibility. By proposing an unsupervised methodology to learn PSD for LBS models (3D animation standard), we overcome both of these drawbacks. Results obtained show cloth-consistency in the animated garments and meaningful pose-dependant folds and wrinkles. Our solution is extremely efficient, handles multiple layers of cloth, allows unsupervised outfit resizing and can be easily applied to any custom 3D avatar.},
        journal = {ACM Trans. Graph.},
        month = {dec},
        articleno = {198},
        numpages = {14},
        keywords = {physics, garment, simulation, deep learning, animation, pose space deformation, neural network}
    }
```
