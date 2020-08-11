# Python repository with some utilities to object detection and tracking
In this repository I store some function for object detection and tracking using Pythin open-cv

# Installation
This repo supports only Python>3.5 and it has been tested only on Linux (Ubuntu) platforms.
We very strongly recommend you install it in a virtual environment. [Here is a quick how-to for Ubuntu](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/). [Conda enviroments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) can also be used.

You can install it by typing the following few steps in a terminal:
1. Clone this repository, `git clone https://github.com/sforazz/detection.git`.
2. cd into the cloned directory, `cd detection`.
3. Create a virtual (or conda) environment. With anaconda you can do it by typing `conda create -n detection python=3.7`.
4. Activate conda environment, `conda activate detection`.
5. Install PyCURT by typing `python setup.py install`

Last step will also create one command, `plot_object_position`. This script will run object tracking followed by a plot of the time the object spent in each position along x direction (the field of view is sampled every 2 cm). The basic setup is for a field of view's width of 110 cm, if your image is different, please change it in scripts/plot_object_position.py (FRAME_LEN_CM).
