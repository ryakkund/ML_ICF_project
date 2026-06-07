# ML_ICF_project
ECE 228 final project Spring 2026 UCSD

We present a Hybrid approach to Deep ONet as a surrogate model for an inertial confinement fusion simulation. Specifically, we use the JAG ICF dataset from Lawrence Livermore that learns hotspot images along with physical quantities such as temperature as a function of relevant experimental input parameters.

The main two files are:
- Deep ONet.py: The actual neural network code that solves the problem. All code is given as functions so that this file doesn't actually output anything.
- JagNet.ipynb: The Jupyter notebook that serves as the driver for the problem and where the outputs and results are shown. This notebook loads the JAG ICF data, calls the functions of DeepONet.py, creates plots of all relevant results.

The three JAG ICF dataset files (to still be pushed) are npy files for the parameters (5 inputs), scalars (15 outputs), and images. They are labeled as such. 
