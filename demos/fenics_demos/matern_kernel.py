import numpy as np
import scipy.linalg as linalg

from dolfin import *
from mshr import *

set_log_level(50)

class source(UserExpression):
    def eval(self,values,x):
        values[0] = 10*np.exp(-(np.power(x[0]-0.5, 2) + np.power(x[1], 2)) )

class matern_cov():
    def __init__(self):
        domain = Circle(Point(0,0),1)
        self.mesh = generate_mesh(domain, 20)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        FEM_el = self.V.ufl_element()
        self.source = source(element=FEM_el)

    def compute_eigen_decomp(self, l=0.2, nu=2):
        self.tau2 = 1/l/l
        self.nu = nu
        a = self.tau2*self.u*self.v*dx - inner( grad(self.u) , grad(self.v) )*dx
        self.L = self.source*self.v*dx

        A, b = assemble_system(a, self.L)
        mat = A.array()

        print('creating kernel ...')
        Ker = np.linalg.matrix_power( mat, -self.nu )

        print('computing eigen decomposition ...')
        v,w = linalg.eig(Ker)
        v = np.real(v)
        w = np.real(w)
        self.l = v
        self.e = w

    def sample(self):
        for i in range(10):
            u = np.random.standard_normal( self.l.shape[0] )
            vec = self.e@(self.l*u) 

            func = Function(self.V)
            func.vector().set_local( vec )
            file = File("sample{}.pvd".format(i))
            file << func

    def save_basis(self, path='./matern_basis.npz'):
        np.savez(path,tau2=self.tau2,nu=self.nu,l=self.l,e=self.e)

    def load_basis(self, path):
        basis_data = np.load(path)
        self.l = basis_data['l']
        self.e = basis_data['e']

if __name__ == '__main__':
    problem = matern_cov()
    problem.compute_eigen_decomp(l=0.2,nu=2)
    problem.save_basis('test_basis.npz')