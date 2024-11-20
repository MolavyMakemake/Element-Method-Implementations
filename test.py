import numpy as np
from hyperbolic.Integrator import Integrator

_int = Integrator(100, open=True)

print(_int.integrate(lambda x, y: np.power(1 - x*x - y*y, -1.5)) - .5 * np.pi)