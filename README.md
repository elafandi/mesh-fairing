# Mesh Fairing
A simple implementation of the algorithm described in "[Triangular surface mesh fairing via Gaussian curvature flow](https://www.sciencedirect.com/science/article/pii/S0377042705004942)" by Huanxi Zhao and Guoliang Xu.

Motivation was to help [the Anthropological and Mathematical Analysis of Archaeological and Zooarchaeological Evidence (AMAAZE)](https://amaaze.umn.edu/) simulate erosion of bone fragments.

Brief description of contents:

* Python files in the main folder
	* `cube_maker.py`: Creates a triangular mesh of the unit cube with desired refinement.
	* `sphere_eroder.py`: Takes in a triangular mesh and generates an animation of its erosion over time, printing the minimum and maximum curvatures (both mean and Gaussian) as it goes.
	* `sphere_eroder_buttons.py`: Like the above, but instead of creating an animation, the 3D graph includes a button for the user to advance the timestep when they're done inspecting the current mesh.
	* `sphere_eroder_charts.py`: Takes in a mesh of a sphere and erodes it as above, generating a pair of graphs at the end: the observed maximum and minimum radius over time compared with the expected radius, and the maximum error in radius over time.
	* `sphere_eroder_mayavi.py`: Legacy inclusion. I think this was a first attempt using a different graphics library. I'm not sure it works.
* `meshes`: A folder of text files in pairs, one in each pair listing the points of a mesh and the other listing the triangles.
	* `tetra`, `cube`, `oct`, `icosa`: Basic meshes of Platonic solids used for testing
	* `sphere_480`, `sphere_7446`, `sphere_30054`: MATLAB-generated meshes of spheres with the listed numbers of nodes
	* `sphere_480_bumpy`: The `sphere_480` mesh with random radial perturbation of the nodes.
* `graphs`:
	* I ran `sphere_eroder_charts` on the `sphere_480` mesh and adjusted the value of beta as defined in the linked paper. The graphs show that the model works well for beta close to 1, but rapidly becomes less accurate as beta increases. Unfortunately, I neglected to include the timestep length when generating these graphs, but by inspection I believe it was small.
	* The `Order` subfolder documents an attempt to find rates of convergence as the mesh becomes more refined and as the timestep shrinks. I have error graphs for all three sphere meshes for timesteps of 1e-2, 1e-3, 1e-4, and (for the 30054 sphere only) 1e-5.
* `animations`:
	* `cube_shrink.gif`: An animation showing a cube gradually get worn down into a sphere. This is what you get when you run `sphere_eroder.py` as it's currently coded.
	* `bump_sphere_shrink.gif`: An animation showing the `sphere_480_bumpy` mesh get worn down until the bumpiness disappears.
	* `bump_sphere_shrink_glitchy.gif`, `bump_sphere_shrink_spiky.gif`, `bump_sphere_shrink_flashy.gif`: Included less for research purposes and more for comedic ones, these animations were produced before I caught bugs in my code that caused the bumpy sphere to grow rather than shrink (the first two) and painted it a different color in each frame (the third one).