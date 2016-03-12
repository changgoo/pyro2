"""
Initialize the doubly periodic shear layer (see, for example, Martin
and Colella, 2000, JCP, 163, 271).  This is run in a unit square
domain, with periodic boundary conditions on all sides.  Here, the
initial velocity is

              / tanh(rho_s (y-0.25))   if y <= 0.5
u(x,y,t=0) = <
              \ tanh(rho_s (0.75-y))   if y > 0.5


v(x,y,t=0) = delta_s sin(2 pi x)


"""

from __future__ import print_function

import math
import numpy as np

import mesh.patch as patch
from util import msg

def init_data(my_data, rp):
    """ initialize the incompressible shear problem """

    msg.bold("initializing the incompressible shear problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in shear.py")

    # get the necessary runtime parameters
    delta_s = rp.get_param("vortex.delta_s")

    print('delta_s = ', delta_s)

    # get the velocities
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    myg = my_data.grid

    ran=1+delta_s*(np.random.rand(myg.qx,myg.qy)-0.5)
    u.d[:,:] = -np.sin(2.0*math.pi*myg.x2d)*np.cos(2.0*math.pi*myg.y2d)*ran
    ran=1+delta_s*(np.random.rand(myg.qx,myg.qy)-0.5)
    v.d[:,:] = np.cos(2.0*math.pi*myg.x2d)*np.sin(2.0*math.pi*myg.y2d)*ran


    print("extrema: ", u.min(), u.max())


def finalize():
    """ print out any information to the user at the end of the run """
    pass
