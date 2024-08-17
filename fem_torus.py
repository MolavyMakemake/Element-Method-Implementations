import dolfinx
import ufl
domain = dolfinx.fem.create_rectangle(1,1)
U = ufl.FunctionSpace(domain, "")