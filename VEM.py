import numpy as np

class Model:
    def __init__(self, vertices=[], polygons=[], trace=[]):
        self.vertices = vertices
        self.polygons = polygons
        self.trace = trace
        self._identify = []

    def basis(self, f):
        n = 0
        elements = []
        elements_mask = []

        for i in range(np.size(self.vertices, axis=1)):
            if i in self.trace:
                elements.append(-1)
            else:
                elements.append(n)
                elements_mask.append(i)
                n += 1

        for i, j in self._identify:
            elements[j] = elements[i]

        L = np.zeros([n, n])
        b = np.zeros(n)

        rot = np.array([[0, 1], [-1, 0]])

        for p_i in self.polygons:
            p1 = self.vertices[:, p_i]
            p0 = np.roll(p1, 1, axis=1)
            p2 = np.roll(p1, -1, axis=1)

            N = np.size(p1, axis=1)

            d0 = np.linalg.norm(p1 - p0, axis=0)
            d1 = np.roll(d0, -1)

            area = 0.5 * np.abs(np.dot(p1[0, :], p0[1, :]) - np.dot(p0[0, :], p1[1, :]))  # shoelace
            circumference = np.sum(d0)

            print("area:", area, ", circumference:", circumference)
            print("points:\n", p1, "\n")

            I_f = area * np.average(f(p1[0, :] + 1j * p1[1, :]))

            proj_D = rot @ (p2 - p0) / (2 * area)
            proj_a = (d0 + d1 - ((p1 + p0) @ d0) @ proj_D) / (2 * circumference)

            for i in range(N):
                e_i = elements[p_i[i]]
                if e_i < 0:
                    continue

                d_ik = i == np.arange(0, N)
                for j in range(N):
                    e_j = elements[p_i[j]]
                    if e_j < 0:
                        continue

                    # compute local LHS
                    d_jk = j == np.arange(0, N)
                    a = np.dot(proj_D[:, i], proj_D[:, j]) * area
                    s = np.sum((d_ik - proj_a[i] - proj_D[:, i] @ p1) * (d_jk - proj_a[j] - proj_D[:, j] @ p1))

                    L[e_i, e_j] += a + s

                # compute local RHS
                b[e_i] += I_f / N

        return L, b, elements_mask

    def solve_poisson(self, f):
        L, b, mask = self.basis(f)

        print(L)
        print(b)

        u = np.zeros(np.size(self.vertices, axis=1), dtype=complex)
        u[mask] = np.linalg.solve(L, b)

        return u

    def solve_spectrum(self):
        L, M, mask = self.basis()
        u = np.zeros(len(self.vertices), dtype=complex)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(L) @ M)

        u[mask] = eigenvectors[0]
        return eigenvalues[0], u / np.sqrt(np.real(np.dot(M @ eigenvectors[0], np.conj(eigenvectors[0]))))