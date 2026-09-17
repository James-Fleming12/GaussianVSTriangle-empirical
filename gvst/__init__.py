"""GaussianVSTriangle: a controlled, synthetic-data comparison of explicit
radiance-field primitives under leave-one-out novel-view evaluation.

Re-implements, in one shared pure-PyTorch rasterization framework:

* 3D Gaussian Splatting (Kerbl et al., 2023)
* 2D Gaussian Splatting (Huang et al., 2024)
* Triangle Splatting (Held et al., 2025)

The goal is *controlled* comparison: identical cameras, identical synthetic
ground truth, identical initialization point cloud, identical optimization
budget and evaluation protocol.  Only the primitive and its paper-specific
regularization differ.
"""

__version__ = "0.1.0"
