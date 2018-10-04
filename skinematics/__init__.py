""""Functions for working with 3D kinematics (i.e. quaternions and rotation
matrices)

Compatible Python 3.

Dependencies
------------
numpy, scipy, matplotlib, pandas, sympy, pygame, pyOpenGL

Homepage
--------
http://work.thaslwanter.at/skinematics/html/

Copyright (c) 2019 Thomas Haslwanter <thomas.haslwanter@fh-ooe.at>

"""

from . import imus, markers, misc, quat, rotmat, vector, view, sensors

__author__ = "Thomas Haslwanter <thomas.haslwanter@fh-linz.at>"
__license__ = "BSD 2-Clause License"
__version__ = "0.8.2"
__all__ = ['imus', 'markers', 'misc', 'quat', 'rotmat',
           'vector', 'view', 'sensors']
