import numpy as np

class Model:
    def __init__(self, vertices=[], triangles=[], trace=[], identify=[]):
        self.vertices = vertices
        self.triangles = triangles
        self.trace = trace
        self.identify = identify

    def basis(self):
        n = 0
        elements = []
        elements_mask = []

        for i in range(len(self.vertices)):
            if i in self.trace:
                elements.append(-1)
            else:
                elements.append(n)
                elements_mask.append(i)
                n += 1

        for i, j in self.identify:
            elements[i] = elements[j]

        M = np.zeros([n, n])
        L = np.zeros([n, n])

        for v_i, v_j, v_k in self.triangles:
            v1 = self.vertices[v_j] - self.vertices[v_i]
            v2 = self.vertices[v_k] - self.vertices[v_i]
            v3 = self.vertices[v_j] - self.vertices[v_k]
            Jac_A = np.abs(v1[0] * v2[1] - v2[0] * v1[1])

            m = Jac_A / 24
            l = .5 / Jac_A

            i, j, k = elements[v_i], elements[v_j], elements[v_k]

            if i >= 0:
                M[i, i] += m * 2
                L[i, i] += l * np.dot(v3, v3)
                if j >= 0:
                    M[(i, j), (j, i)] += m
                    L[(i, j), (j, i)] += l * np.dot(v3, v2)
                if k >= 0:
                    M[(i, k), (k, i)] += m
                    L[(i, k), (k, i)] -= l * np.dot(v3, v1)

            if j >= 0:
                M[j, j] += m * 2
                L[j, j] += l * np.dot(v2, v2)
                if k >= 0:
                    M[(j, k), (k, j)] += m
                    L[(j, k), (k, j)] -= l * np.dot(v1, v2)

            if k >= 0:
                M[k, k] += m * 2
                L[k, k] += l * np.dot(v1, v1)

        return M, L, elements_mask

    def solve_poisson(self, f):
        M, L, mask = self.basis()
        u = np.zeros(len(self.vertices), dtype=complex)
        b = np.zeros(len(self.vertices), dtype=complex)

        b[mask] = M @ f(self.vertices)
        u[mask] = np.linalg.solve(L, b)

        return u

    def solve_spectrum(self):
        L, M, mask = self.basis()
        u = np.zeros(len(self.vertices), dtype=complex)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(L) @ M)

        u[mask] = eigenvectors[0]
        return eigenvalues[0], u