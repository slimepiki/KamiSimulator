# To-do

Although I won't update this program, I leave the to-do list here to tell what is unimplemented.

## Solver

Implementing

- [Position Based Discrete Elastic Rods]( https://dl.acm.org/doi/10.5555/2849517.2849522 )
- [Selle's mass-spring system]( https://dl.acm.org/doi/10.1145/1360612.1360663 ), or [Real-time hair simulation with heptadiagonal decomposition on mass spring system]( https://www.sciencedirect.com/science/article/abs/pii/S1524070320300217 )
- and, Lagrangian collision resolver

## Body::Fullbody

I assume to use [SMPL](https://smpl.is.tue.mpg.de/).

Modifying

- ```Body::Confirm()```
- and, ```Body::GenerateSDF()``` (SDF won't be suitable for full-body because of its shape...).

## Bone Animation

I assume to use [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/).

Defining

- ```Animation mesh``` class to manage the mesh info,
- ```struct keyFrame``` to pass the current posture
- and, ```Motion``` class to manage the animation data.

Implementing

- CMU Motion Capture importer in ```importer``` directory,
- ```Motion::Motion()```,
- ```keyFrame Motion::GetFrme(float time)```
- and, ```Body::SetKeyFrame(keyFrame k)``` to deform the body to the current posture.

## Stable Cosserat Rods Solver

- Making ```KittenEngine/KittenGpuLBVH/lbvh```  thrust-free to account for the huge size collision queries. The thrust library seems not to handle buffer allocation of more than about 1GB in size.
