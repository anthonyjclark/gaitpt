# Gaitpt

Pre-training gaits for genetic algorithms

# Experiments

- Initial leg configuration
	+ Biological (back femur angled forward) (this one)
	+ Robotic (like spot)

- Forward/backward at constant speed
	+ How would we do otherwise with sinusoid input

- Sliders might work with IK

# Initial Notes

- Walking pattern generator
	+ gaits: walk (& amble), trot, pace (rack), canter, and gallop
	+ other: crawl
	+ canter and gallop are asymmetrical
	+ descriptions of gaits: https://www.animatornotebook.com/learn/quadrupeds-gaits
	+ gait animations: https://mymodernmet.com/animal-gaits-animation-stephen-cunnane/
- Inverse kinematics
	+ get intermediate angles? (less robotic)
		* Avoid obstacles
		* Efficiency
- In the loop

let evolution de-robotify the gait


IK resources:
- [Inverse Kinematics - Sublucid Geometry](https://zalo.github.io/blog/inverse-kinematics/ "Inverse Kinematics - Sublucid Geometry")
- [zalo/MathUtilities: A collection of some of the neat math and physics tricks that I've collected over the last few years.](https://github.com/zalo/MathUtilities "zalo/MathUtilities: A collection of some of the neat math and physics tricks that I've collected over the last few years.")
- [(PDF) FABRIK: A fast, iterative solver for the Inverse Kinematics problem](https://www.researchgate.net/publication/220632147_FABRIK_A_fast_iterative_solver_for_the_Inverse_Kinematics_problem "(PDF) FABRIK: A fast, iterative solver for the Inverse Kinematics problem")
- [Cyclic Coordinate Descent in 2D - RyanJuckett.com](https://www.ryanjuckett.com/cyclic-coordinate-descent-in-2d/ "Cyclic Coordinate Descent in 2D - RyanJuckett.com")
- [Cyclic Coordonate descent Inverse Kynematic (CCD IK) - Rodolphe Vaillant's homepage](http://rodolphe-vaillant.fr/?e=114 "Cyclic Coordonate descent Inverse Kynematic (CCD IK) - Rodolphe Vaillant's homepage")

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
