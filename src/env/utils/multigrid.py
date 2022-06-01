'''See https://www.mathematik.uni-wuerzburg.de/fileadmin/10040900/2019/mgintro.pdf
- multigrid method : use hierarchy of discretizations
- OpenMG : A new multigrid implementation in python, http://conference.scipy.org/proceedings/scipy2012/pdfs/tom_bertalan.pdf
'''
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.sparse import eye
from typing import Optional, Union, Tuple, List

def restrict(origin, out = None, avg = False):
    ''' coarsen the original matrix onto a coarser mesh
    inputs : origin (nx x ny 2D array)
    '''

    nx = origin.shape[0]
    ny = origin.shape[1]

    if (nx - 1) % 2 == 1 or (ny - 1) % 2 == 1:
        if out is None:
            return origin
        out.resize(origin.shape)
        out[:,:] = origin
        return
    
    nx = (nx - 1) // 2 + 1
    ny = (ny - 1) // 2 + 1

    if out is None:
        out = np.zeros((nx, ny))
    else:
        out.resize(nx,ny)
    
    for idx_x in range(1, nx - 1):
        for idx_y in range(1, ny - 1):
            idx_x0 = 2 * idx_x
            idx_y0 = 2 * idx_y
            
            out[idx_x, idx_y] = origin[idx_x0, idx_y0] / 4.0 
            + (origin[idx_x0 + 1, idx_y0] + origin[idx_x0 - 1, idx_y0] + origin[idx_x0, idx_y0 + 1] + origin[idx_x0, idx_y0 - 1]) / 8.0
            + (origin[idx_x0 + 1, idx_y0 + 1] + origin[idx_x0 - 1, idx_y0 + 1] + origin[idx_x0 - 1, idx_y0 + 1] + origin[idx_x0 - 1, idx_y0 - 1]) / 16.0

    if not avg:
        out *= 4.0

    return out

def interpolate(origin, out = None):
    '''Interpolate a solution onto a finer mesh
    '''

    nx = origin.shape[0]
    ny = origin.shape[1]

    nx2 = 2 * (nx - 1) + 1
    ny2 = 2 * (ny - 1) + 1

    if out is None:
        out = np.zeros((nx2, ny2))
    else:
        out[:,:] = 0

    for idx_x in range(1, nx - 1):
        for idx_y in range(1, ny - 1):
            idx_x0 = 2 * idx_x
            idx_y0 = 2 * idx_y
            
            out[idx_x0 - 1, idx_y0 - 1] += 0.25 * origin[idx_x, idx_y]
            out[idx_x0 - 1, idx_y0] += 0.5 * origin[idx_x, idx_y]
            out[idx_x0 - 1, idx_y0 + 1] += 0.25 * origin[idx_x, idx_y]

            out[idx_x0, idx_y0 - 1] += 0.5 * origin[idx_x, idx_y]
            out[idx_x0, idx_y0] = origin[idx_x, idx_y]
            out[idx_x0, idx_y0 + 1] += 0.5 * origin[idx_x, idx_y]

            out[idx_x0 + 1, idx_y0 - 1] += 0.25 * origin[idx_x, idx_y]
            out[idx_x0 + 1, idx_y0] += 0.5 * origin[idx_x, idx_y]
            out[idx_x0 + 1, idx_y0 + 1] += 0.25 * origin[idx_x, idx_y]
    
    return out

class SimpleMultigrid:
    def __init__(self, A):
        self.solve = factorized(A.tocsc())

    def __call__(self, x, b):
        b1d = np.reshape(b, -1)
        x = self.solve(b1d)

        return np.reshape(x, b.shape)
    
class MultigridJacobi:
    def __init__(self, A, n_cycle : int = 4, n_iter : int = 10, subsolver = None):
        self.A = A
        self.diag = A.diagonal()
        self.subsolver = subsolver
        self.n_iter = n_iter
        self.n_cycle = n_cycle

        self.sub_b = None
        self.x_update = None
    
    def __call__(self, xi, bi, n_cycle = None, n_iter = None):
        x = np.reshape(xi, -1)
        b = np.reshape(bi, -1)

        if n_cycle is None:
            n_cycle = self.n_cycle
        
        if n_iter is None:
            n_iter = self.n_iter
        
        for cycle in range(n_cycle):
            for i in range(n_iter):
                x += (b - self.A.dot(x)) / self.diag

            if self.subsolver:
                error = b - self.A.dot(x)
                
                self.sub_b = restrict(np.reshape(error, xi.shape))
            
                sub_x = np.zeros(self.sub_b.shape)
                sub_x = self.subsolver(sub_x, self.sub_b)

                self.x_update = interpolate(sub_x)

                x += np.reshape(self.x_update, -1)

            for i in range(n_iter):
                x += (b - self.A.dot(x)) / self.diag
            
        return x.reshape(xi.shape)

def createVcycle(nx : int, ny : int, generator, n_levels : int = 4, n_cycle : int = 1, n_iter : int = 10, direct : bool = True):
    if (nx - 1) % 2 == 1 or (ny - 1) % 2 == 1:
        n_levels = 1
    
    if n_levels > 1:
        nxsub = (nx - 1) // 2 + 1
        nysub = (ny - 1) // 2 + 1

        subsolver = createVcycle(nxsub, nysub, generator, n_levels - 1, n_iter, direct)

        A = generator(nx, ny)

        return MultigridJacobi(A, n_cycle = n_cycle, n_iter = n_iter, subsolver=subsolver)

    A = generator(nx, ny)

    if direct:
        return SimpleMultigrid(A)
    
    return MultigridJacobi(A, n_iter = n_iter, n_cycle=n_cycle, subsolver=subsolver)
        
def smoothJacobi(A, x:Union[np.ndarray, np.array], b:Union[np.ndarray, np.array], dx : float, dy:float):
    '''smooth the solution using jacobi method
    '''

    if b.shape != x.shape:
        raise ValueError("b and x have different shapes")
    
    smooth = x + (b - A(x,dx,dy)) / A.diag(dx,dy)

    return smooth

def smoothVcycle(A,x,b,dx,dy,n_iter = 10, sublevels = 0, direct = True):
    for i in range(n_iter):
        x = smoothJacobi(A,x,b,dx,dy)
    
    if sublevels > 0:
        error = b - A(x,dx,dy)
    
        Cerror = restrict(error)

        Cx = np.zeros(Cerror.shape)
        Cx = smoothVcycle(A,Cx,Cerror, dx * 2.0, dy * 2.0, n_iter, sublevels - 1, True)
    
        x_update = interpolate(Cx)

        x += x_update
    
    for i in range(n_iter):
        x = smoothJacobi(A,x,b,dx,dy)
    
    return x

def smoothMG(A,x,b,dx,dy,n_iter = 10, sublevels = 1, n_cycle = 2):
    error = b - A(x, dx, dy)
    print("Starting max residual: %e" % (np.max(abs(error)),))

    for c in range(n_cycle):
        x = smoothVcycle(A, x, b, dx, dy, n_iter, sublevels)

        error = b - A(x, dx, dy)
        print("Cycle %d : %e"%(c,np.max(abs(error))))
    return x

class LaplacianOp:
    """
    Implements a simple Laplacian operator
    for use with the multigrid solver
    """

    def __call__(self, f, dx, dy):
        nx = f.shape[0]
        ny = f.shape[1]

        b = np.zeros([nx, ny])

        for x in range(1, nx - 1):
            for y in range(1, ny - 1):
                # Loop over points in the domain

                b[x, y] = (f[x - 1, y] - 2 * f[x, y] + f[x + 1, y]) / dx ** 2 + (
                    f[x, y - 1] - 2 * f[x, y] + f[x, y + 1]
                ) / dy ** 2

        return b

    def diag(self, dx, dy):
        return -2.0 / dx ** 2 - 2.0 / dy ** 2


class LaplaceSparse:
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly

    def __call__(self, nx, ny):
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)

        # Create a linked list sparse matrix
        N = nx * ny
        A = eye(N, format="lil")
        for x in range(1, nx - 1):
            for y in range(1, ny - 1):
                row = x * ny + y
                A[row, row] = -2.0 / dx ** 2 - 2.0 / dy ** 2

                # y-1
                A[row, row - 1] = 1.0 / dy ** 2

                # y+1
                A[row, row + 1] = 1.0 / dy ** 2

                # x-1
                A[row, row - ny] = 1.0 / dx ** 2

                # x+1
                A[row, row + ny] = 1.0 / dx ** 2
        # Convert to Compressed Sparse Row (CSR) format
        return A.tocsr()


if __name__ == "__main__":

    # Test case

    from numpy import meshgrid, exp, linspace
    import matplotlib.pyplot as plt

    from timeit import default_timer as timer

    nx = 65
    ny = 65

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    xx, yy = meshgrid(linspace(0, 1, nx), linspace(0, 1, ny))

    rhs = exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4 ** 2)

    rhs[0, :] = 0.0
    rhs[:, 0] = 0.0
    rhs[nx - 1, :] = 0.0
    rhs[:, ny - 1] = 0.0

    x = np.zeros([nx, ny])

    x2 = x.copy()

    A = LaplacianOp()

    ################ SIMPLE ITERATIVE SOLVER ##############

    for i in range(1):
        x2 = smoothJacobi(A, x, rhs, dx, dy)
        x, x2 = x2, x  # Swap arrays
        error = rhs - A(x, dx, dy)
        print("%d : %e"%(i, np.max(abs(error))))

    ################ MULTIGRID SOLVER #######################

    print("Python multigrid solver")

    x = np.zeros([nx, ny])

    start = timer()
    x = smoothMG(A, x, rhs, dx, dy, n_iter=5, sublevels=3, n_cycle=2)
    end = timer()

    error = rhs - A(x, dx, dy)
    print("Max error : {0}".format(np.max(abs(error))))
    print("Run time  : {0} seconds".format(end - start))

    ################ SPARSE MATRIX ##########################

    print("Sparse matrix solver")

    x2 = np.zeros([nx, ny])

    start = timer()
    solver = createVcycle(
        nx, ny, LaplaceSparse(1.0, 1.0), n_cycle=2, n_iter=5, n_levels=4, direct=True
    )

    start_solve = timer()
    x2 = solver(x2, rhs)

    end = timer()

    error = rhs - A(x2, dx, dy)
    print("Max error : {0}".format(np.max(abs(error))))
    print(
        "Setup time: {0}, run time: {1} seconds".format(
            start_solve - start, end - start_solve
        )
    )

    print("Values: {0}, {1}".format(x2[10, 20], x[10, 20]))

    f = plt.figure()
    # plt.contourf(x)
    plt.plot(x[:, 32])
    plt.plot(x2[:, 32])
    plt.show()