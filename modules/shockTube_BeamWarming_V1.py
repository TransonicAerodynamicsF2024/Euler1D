#   Purpose:
#       - This script contains the class definition for the ShockTube class.
#
#   Record of Revision:
#       Date            Programmer              Description
#       ====            ==========              ===========
#       15 Oct, 2024    Paramvir Lobana         Original Code

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.linalg import solve_banded

# read and process data from csv file
# TODO - the script should be able to recognize the system(unix or windows) 
#        and read the file accordingly.

df = pd.read_csv('modules/sod-exact_solution.txt')

def gauss_seidel(M, RHS, Q_init, max_iters=1000, tol=1e-6, relaxation=0.8):
    """
    Perform Gauss-Seidel iteration to solve M * Q_new = RHS with relaxation.

    Parameters:
    M         -- Block tridiagonal coefficient matrix.
    RHS       -- Right-hand side vector.
    Q_init    -- Initial guess for the solution.
    max_iters -- Maximum number of iterations.
    tol       -- Convergence tolerance.
    relaxation -- Under-relaxation factor (default is 0.8).

    Returns:
    Q_new -- Solution vector after Gauss-Seidel iterations.
    """
    Q_new = Q_init.copy()
    Nx = len(RHS) // 3  # Adjust for block size
    epsilon = 1e-10     # Small constant to prevent division by zero

    for iter_num in range(max_iters):
        Q_old = Q_new.copy()

        # Update each block row
        for i in range(Nx):
            row_start = 3 * i  # Block size is 3 for Q
            row_end = row_start + 3

            # Extract diagonal, upper, and lower blocks
            diag_block = M[row_start:row_end, row_start:row_end]
            RHS_block = RHS[row_start:row_end]
            
            # Apply bounds to avoid overflow in lower and upper matrix operations
            if i > 0:
                lower_block = M[row_start:row_end, row_start - 3:row_start]
                RHS_block -= lower_block @ np.clip(Q_new[row_start - 3:row_start], -1e10, 1e10)
                
            if i < Nx - 1:
                upper_block = M[row_start:row_end, row_end:row_end + 3]
                RHS_block -= upper_block @ np.clip(Q_new[row_end:row_end + 3], -1e10, 1e10)
            
            # Solve for the current block and apply relaxation
            new_value = np.linalg.solve(diag_block, RHS_block)
            Q_new[row_start:row_end] = relaxation * new_value + (1 - relaxation) * Q_old[row_start:row_end]

            # Apply clipping to stabilize intermediate values
            Q_new[row_start:row_end] = np.clip(Q_new[row_start:row_end], -1e10, 1e10)

        # Check convergence
        residual = np.linalg.norm(Q_new - Q_old)
        if residual < tol:
            break

    return Q_new



@dataclass
class FluidProperties:
    gamma: float = 1.4  # ratio of specific heat for ideal gas
    epsilon_e: float = 0.1  # artificial dissipation parameter
    epsilon_i: float = epsilon_e * 2.5  # artificial dissipation parameter


class shockTube_BW:
    def __init__(self, nx:int=1000, length:float=1000.0, gamma:float=1.4) -> None:
        self.nx = nx
        self.dx = length / nx
        self.gamma = gamma
        self.fluid = FluidProperties(gamma=gamma)

        self.epsilon_e = self.fluid.epsilon_e
        self.epsilon_i = self.fluid.epsilon_i

        # Initialize the arrays for conservative variables
        self.rho = np.zeros(nx)
        self.rhou = np.zeros(nx)
        self.e = np.zeros(nx) # energy
        self.u = np.zeros(nx)
        self.p = np.zeros(nx)

        self.setupLogging()

    def setupLogging(self):
        # TODO - change the logging function for beam warming method
        # configure the logging to write results to console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def  setInitialConditions(self):
        x = np.linspace(0, self.nx * self.dx, self.nx)

        # Define left side -> low pressure, low density
        leftSide = x < 500
        self.rho[leftSide] = 1.0
        self.p[leftSide] = 1.0

        # Define left side -> low pressure, low density
        rightSide = x >= 500
        self.rho[rightSide] = 4.0
        self.p[rightSide] = 4.0

        self.u[:] = 0.0
        self.e[:] = self.p[:] / (self.gamma - 1.0) + 0.5 * self.rho[:] * self.u[:]**2
        self.rhou[:] = self.rho[:] * self.u[:]

    def computeFlux(self, rho, rhou, e):
        u = rhou / rho
        p = (self.gamma - 1.0) * (e - 0.5 * rho * u**2)
        
        flux_rho = rhou
        flux_rhou = rhou * u + p
        flux_e = (e + p) * u
        return np.array([flux_rho, flux_rhou, flux_e])

    def computeJacobian(self, i):

        """
        Compute the 3x3 Jacobian matrix A(Q) for the quasi-1D Euler equations.
        
        Parameters:
        Q     -- State vector at a grid point [rho*A, rho*u*A, e*A].
        gamma -- Specific heat ratio (default is 1.4 for air).
        
        Returns:
        A -- 3x3 Jacobian matrix evaluated at the given state vector Q.
        """
        # Construct the Jacobian matrix
        A = np.zeros((3, 3))
        A[0, 1] = 1  # d(rho*u*A)/dQ

        A[1, 0] = (self.gamma - 3) / 2 * self.u[i]**2
        A[1, 1] = (3 - self.gamma) * self.u[i]
        A[1, 2] = self.gamma - 1

        A[2, 0] = (self.gamma - 1) / 2 * self.u[i]**3 - self.u[i] * self.e[i]
        A[2, 1] = self.gamma * self.e[i] - (self.gamma - 1) * self.u[i]**2
        A[2, 2] = self.gamma * self.u[i]

        return A
    
    def applyBoundaryCond(self):
        # Apply zero-order boundary conditions
        # Set the first and last points to match their nearest neighbors
        self.rho[0] = self.rho[1]
        self.rho[-1] = self.rho[-2]

        self.rhou[0] = self.rhou[1]
        self.rhou[-1] = self.rhou[-2]

        self.e[0] = self.e[1]
        self.e[-1] = self.e[-2]

    def calculateTimeStep(self, CFL):
        """
        Calculate the time step based on the CFL condition.
        """
        c = np.sqrt(self.gamma * self.p / self.rho)
        dt = CFL * self.dx / (np.max(np.abs(self.u) + c))
        return dt
 

    def blockTridiagonal(self, dt):
        """
        Construct a block tridiagonal coefficient matrix for the Beam-Warming method.

        Parameters:
        Q         -- State vector array (Nx x 3) containing [rho*A, rho*u*A, e*A] at each grid point.
        Nx        -- Number of spatial grid points.
        dx        -- Grid spacing.
        dt        -- Time step size.
        epsilon_i -- Implicit dissipation coefficient.
        gamma     -- Specific heat ratio (default is 1.4).

        Returns:
        M -- Block tridiagonal matrix of size (3*Nx, 3*Nx).
        """
        block_size = 3  # Size of each block (since Q has 3 components)
        M = np.zeros((block_size * self.nx, block_size * self.nx))

        I = np.eye(block_size)  # Identity matrix for each block

        for i in range(self.nx):
            A_i = self.computeJacobian(i)  # Jacobian at grid point i

            # Main diagonal block
            main_block = I - 2 * (self.epsilon_i / self.dx**2) * I
            M[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = main_block

            if i < self.nx - 1:
                # Upper diagonal block
                upper_block = (self.epsilon_i / self.dx**2) * I + (dt / (2 * self.dx)) * A_i
                M[i*block_size:(i+1)*block_size, (i+1)*block_size:(i+2)*block_size] = upper_block

                # Lower diagonal block
                lower_block = (self.epsilon_i / self.dx**2) * I - (dt / (2 * self.dx)) * A_i
                M[(i+1)*block_size:(i+2)*block_size, i*block_size:(i+1)*block_size] = lower_block

        return M
    
    def computeRHS(self, Q, dt):
        rhs = np.zeros_like(Q)

        # Flux vector components
        E = np.zeros_like(Q)
        E[0, :] = self.rho  # Mass flux
        E[1, :] = self.rho * self.u**2 + self.p  # Momentum flux
        E[2, :] = self.u * (self.e + self.p)  # Energy flux

        # 1. Explicit Flux Gradient Term: - dt * (dE/dx)
        flux_gradient = np.zeros_like(Q)
        for i in range(1, self.nx - 1):
            flux_gradient[:, i] = (E[:, i + 1] - E[:, i]) / self.dx
        rhs -= dt * flux_gradient

        # 2. Explicit Dissipation Term: - epsilon_e * (d^2Q/dx^2)
        dissipation = np.zeros_like(Q)
        for i in range(1, self.nx - 1):
            dissipation[:, i] = (Q[:, i + 1] - 2 * Q[:, i] + Q[:, i - 1]) / self.dx**2
        rhs -= self.epsilon_e * dissipation

        return rhs
    

    def timeMarchBeamWarming(self, t_final, dt, CFL=0.5, max_iters=1000, tol=1e-6):
        """
        Perform time marching using the implicit Beam-Warming method with Gauss-Seidel solver.
        
        Parameters:
        t_final -- Final time.
        dt      -- Time step size.
        """
        n_steps = int(t_final / dt)

        for step in range(n_steps):
            # Update dt dynamically using the CFL condition
            dt = self.calculateTimeStep(CFL)

            # Apply boundary conditions before computing RHS and M matrix
            self.applyBoundaryCond()

            # Construct the block tridiagonal matrix M
            M = self.blockTridiagonal(dt)

            # Compute the RHS
            Q = np.array([self.rho, self.rhou, self.e])
            rhs = self.computeRHS(Q, dt)
            RHS = Q.flatten() + rhs.flatten()

            # Initial guess for Gauss-Seidel
            Q_init = Q.flatten()

            # Solve the linear system M * Q_new = RHS using Gauss-Seidel
            Q_new = gauss_seidel(M, RHS, Q_init, max_iters=max_iters, tol=tol)

            # Reshape Q_new back to (3 x nx) for the next iteration
            Q_new = Q_new.reshape(3, self.nx)

            # Update solution and apply boundary conditions after updating
            self.rho = Q_new[0]
            self.rhou = Q_new[1]
            self.e = Q_new[2]
            self.applyBoundaryCond()

            # Update primitive variables with a small epsilon to avoid division by zero
            epsilon = 1e-10
            self.u = self.rhou / (self.rho + epsilon)
            self.p = (self.gamma - 1.0) * (self.e - 0.5 * self.rho * self.u**2)

            # Log progress every 10 steps
            if step % 10 == 0:
                self.logger.info(f"Step {step}/{n_steps} completed.")

        return self.rho, self.u, self.p


    def plotResults(self, showExact=False, showInitial=True):
        x = np.linspace(0, self.nx * self.dx, self.nx)
        initialLinewidth = 1.0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        # Plot density
        if showInitial:
            ax1.plot([0, 500, 500, 1000], [1, 1, 4, 4], 'r--', label='Initial', linewidth=initialLinewidth)
        if showExact:
            ax1.plot(df.x, df.rho, 'g-', label='Exact', linewidth=0.75)
        ax1.plot(x, self.rho, 'b-', label='t = 250')

        ax1.set_ylabel(r'$\rho$')
        ax1.set_xlabel(r'$x$')
        ax1.grid(True)
        ax1.legend()
        
        # Plot velocity
        if showInitial:
            ax2.plot([0, 1000], [0, 0], 'r--', label='Initial', linewidth=initialLinewidth)
        if showExact:
            ax2.plot(df.x, df.u, 'g-', label='Exact', linewidth=0.75)
        ax2.plot(x, self.u, 'b-', label='t = 250')

        ax2.set_ylabel(r'$u$')
        ax2.set_xlabel(r'$x$')
        ax2.grid(True)
        ax2.legend()
        
        # Plot pressure
        if showInitial:
            ax3.plot([0, 500, 500, 1000], [1, 1, 4, 4], 'r--', label='Initial', linewidth=initialLinewidth)
        if showExact:
            ax3.plot(df.x, df.p, 'g-', label='Exact', linewidth=0.75)
        ax3.plot(x, self.p, 'b-', label='t = 250')

        ax3.set_ylabel(r'$p$')
        ax3.set_xlabel(r'$x$')
        ax3.grid(True)
        ax3.legend()
        plt.tight_layout()
        plt.show()
