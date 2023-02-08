# Gaitpt

Pre-training gaits for genetic algorithms

![](example.gifj)

# Process

1. Create robot and gait patterns (e.g., `dog_config.json`)
2. Run `gaitpt.py` to generate kinematics data in `KinematicsData`
	- saves animations in `Animations`
3. Run `gait_model.ipynb` to generate trained models in `Models`
	- saves model outputs in `ModelOutputs`
	- saves figures used for comparisons in `Figures`

# Initial Notes

- Walking pattern generator
	+ gaits: walk (& amble), trot, pace (rack), canter, and gallop
	+ other: crawl
	+ canter and gallop are asymmetrical
	+ [A GUIDE TO QUADRUPEDS’ GAITS - Walk, amble, trot, pace, canter, gallop — Animator Notebook](https://www.animatornotebook.com/learn/quadrupeds-gaits)
	+ [Learn About Animal Gaits With This Clever Animation](https://mymodernmet.com/animal-gaits-animation-stephen-cunnane)
- Inverse kinematics
	+ get intermediate angles? (less robotic)
		* Avoid obstacles
		* Efficiency
- In the evolutionary loop
- Let evolution de-robotify the gait
- Initial leg configuration
	+ Biological (back femur angled forward) (this one)
	+ Robotic (like spot)
- Forward/backward at constant speed
	+ How would we do otherwise with sinusoid input
- Sliders might work with IK

# IK Resources

- [Inverse Kinematics - Sublucid Geometry](https://zalo.github.io/blog/inverse-kinematics)
- [zalo/MathUtilities: A collection of some of the neat math and physics tricks that I've collected over the last few years.](https://github.com/zalo/MathUtilities)
- [(PDF) FABRIK: A fast, iterative solver for the Inverse Kinematics problem](https://www.researchgate.net/publication/220632147_FABRIK_A_fast_iterative_solver_for_the_Inverse_Kinematics_problem)
- [Cyclic Coordinate Descent in 2D - RyanJuckett.com](https://www.ryanjuckett.com/cyclic-coordinate-descent-in-2d)
- [Cyclic Coordonate descent Inverse Kynematic (CCD IK) - Rodolphe Vaillant's homepage](http://rodolphe-vaillant.fr/?e=114)

# Design Notes

## Top-Down Design

1. Animat (fixed at four legs)
	1. Gaits (standalone file with patterns)
	2. Maintains four legs
		1. Compute angles of each joint using CCD IK
		2. Creates animation actors
	3. Directs tip/foot of each leg based on gaits
	4. Coordinates animation (updates each leg)
		+ axis
		+ matplotlib stuff

## TODO

- Remove Animation/*_kinematic.gif?
