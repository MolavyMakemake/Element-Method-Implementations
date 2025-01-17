import numpy as np
from Integrator import Integrator
import matplotlib.pyplot as plt

dVol = lambda x, y: np.power(1 - x * x - y * y, -1.5)

d1f = lambda x, y: np.power(1 - y*y, .25) * np.power((1 - x * x) * \
                     (1 - x * x - y * y), -1.25) * (1 - x*x - y*y + .5 * x*x * y*y)
d2f = lambda x, y: x*x*x * y * np.power(1 - y*y, -.75) * \
                    np.power(1 - x*x, -.25) * np.power(1 - x * x - y * y, -1.25)

f = lambda x, y: (1 - x*x - y*y) * (d1f(x, y) * ((1 - x*x) * d1f(x, y) + x*y * d2f(x, y)) + \
                                d2f(x, y) * (x*y * d1f(x, y) + (1 - y*y) * d2f(x, y)))

fX_dx = lambda x, y: -f(x, y) * y * (1 - y * y / (1 - x * x)) * dVol(x, y)
fX_dy = lambda x, y: f(x, y) * x * (1 - x * x / (1 - y * y)) * dVol(x, y)

u = np.array([-.5, .8])
v = np.array([-.3, 0.0])
w = np.array([.5, .2])

N_samples = []
I = []
J = []

for N in [100, 200, 300, 400]:
    print("N =",N)

    N_t = N * (N + 1) // 6
    dt = 1 / N
    t = np.linspace(0, 1, N+1)

    F1 = lambda x, y: u[0] + (v[0] - u[0]) * x + (w[0] - u[0]) * y
    F2 = lambda x, y: u[1] + (v[1] - u[1]) * x + (w[1] - u[1]) * y

    path_uv = (u[0] + t * (v[0] - u[0]), u[1] + t * (v[1] - u[1]))
    path_vw = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    path_wu = (w[0] + t * (u[0] - w[0]), w[1] + t * (u[1] - w[1]))

    _int = Integrator(N)
    _I = _int.integrate(lambda x, y: f(F1(x, y), F2(x, y)) * dVol(F1(x, y), F2(x, y)))
    _I *= np.abs((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))

    _J_v = (v[0] - u[0]) * fX_dx(*path_uv) + (v[1] - u[1]) * fX_dy(*path_uv) \
        + (w[0] - v[0]) * fX_dx(*path_vw) + (w[1] - v[1]) * fX_dy(*path_vw) \
        + (u[0] - w[0]) * fX_dx(*path_wu) + (u[1] - w[1]) * fX_dy(*path_wu)

    _J = np.sum(_J_v * dt) - .5 * (_J_v[0] + _J_v[-1]) * dt

    N_samples.append(N * (N + 1) // 2)
    I.append(_I)
    J.append(_J / 2.2)

plt.plot(N_samples, I, "o--", label="interior integral")
plt.plot(N_samples, J, "o--", label="path integral")

plt.xlabel("nr. samples")
plt.ylabel("out")
plt.legend()

plt.show()
