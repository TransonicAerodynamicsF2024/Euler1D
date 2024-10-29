#   Purpose:
#       - This script contains the class definition for the ShockTube class.
#
#   Record of Revision:
#       Date            Programmer              Description
#       ====            ==========              ===========
#       21 Oct, 2024    Paramvir Lobana         Original Code

import numpy as np
import pandas as pd
import logging
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from dataclasses import dataclass

# read and process data from csv file
# TODO - the script should be able to recognize the system(unix or windows) 
#        and read the file accordingly.

df = pd.read_csv('modules/exact.csv')
df['p'] = df['p'].values[::-1]

@dataclass
class FluidProperties:
    gamma: float = 1.4

class shockTube_BeamWarming:
    def __init__(self, 
                 length=1000.0, 
                 nx=201, 
                 t_max=250.0,
                 gamma=1.4,
                 cfl = 0.25) -> None:
        
        self.length = length       
        self.nx = nx               
        self.dx = length / (nx - 1)
        self.t_max = t_max         
        self.gamma = gamma         
        self.cfl = cfl

        self.dt = self.cfl  * self.dx / np.sqrt(gamma * 4.0 / 1.0)
        self.nt = int(self.t_max / self.dt)

        self.epsilon_e = 0.1
        self.epsilon_i = 2.5 * self.epsilon_e
        self.theta = 0.5           # Implicit scheme

        self.x = np.linspace(0, self.length, self.nx)
        self.Q = np.zeros((self.nx, 3))

        # Initiate some functions when class is called
        self.setupLogging()
        self.updatePlotSettings()

    def setupLogging(self):
        # TODO - change the logging function for beam warming method
        # configure the logging to write results to console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def setInitialConditions(self):
        for i in range(self.nx):
            if self.x[i] < self.length / 2:
                rho = 1.0
                p = 1.0
                u = 0.0
            else:
                rho = 4.0
                p = 4.0
                u = 0.0
            e = p / (self.gamma - 1) + 0.5 * rho * u ** 2
            self.Q[i, 0] = rho
            self.Q[i, 1] = rho * u
            self.Q[i, 2] = e

    def computePressure(self, Q):
        rho = Q[:, 0]
        rho_u = Q[:, 1]
        e = Q[:, 2]
        u = rho_u / rho
        p = (self.gamma - 1) * (e - 0.5 * rho * u ** 2)
        return p

    def computeFlux(self, Q):
        rho = Q[:, 0]
        rho_u = Q[:, 1]
        e = Q[:, 2]
        u = rho_u / rho
        p = self.computePressure(Q)
        F = np.zeros_like(Q)
        F[:, 0] = rho_u
        F[:, 1] = rho_u * u + p
        F[:, 2] = u * (e + p)
        return F

    def computeJacobian(self, Q):
        """
        NOTE: Reference from textbook
        The Jacobian matrix A is constructed at all points for the implicit
        steps and, since it operates on a 1st-derivative central difference operator,
        it is computed for the lower and the upper band of the block diagonal matrix.
        """
        rho = Q[:, 0]
        rho_u = Q[:, 1]
        e = Q[:, 2]
        u = rho_u / rho
        p = self.computePressure(Q)
        H = (e + p) / rho

        A = np.zeros((self.nx, 3, 3))
        for i in range(self.nx):
            A[i, 0, 0] = 0.0
            A[i, 0, 1] = 1.0
            A[i, 0, 2] = 0.0

            A[i, 1, 0] = 0.5 * (self.gamma - 3) * u[i] ** 2
            A[i, 1, 1] = (3 - self.gamma) * u[i]
            A[i, 1, 2] = self.gamma - 1

            A[i, 2, 0] = u[i] * (0.5 * (self.gamma - 1) * u[i] ** 2 - H[i])
            A[i, 2, 1] = H[i] - (self.gamma - 1) * u[i] ** 2
            A[i, 2, 2] = self.gamma * u[i]

        return A

    def applyBoundaryConditions(self, Q):
        Q[0, :] = Q[1, :]
        Q[-1, :] = Q[-2, :]
        return Q

    def beamWarmingStep(self, Q):
        nx = self.nx
        dx = self.dx
        dt = self.dt
        theta = self.theta
        epsilon_e = self.epsilon_e
        epsilon_i = self.epsilon_i
        
        # Step -> 1
        # Compute Jacobian matrices and fluxes
        A_matrices = self.computeJacobian(Q)
        F = self.computeFlux(Q)

        # Initialize LHS and RHS of the Beam-Warming Equation
        size = nx * 3  # Total number of unknowns
        data = []
        rows = []
        cols = []
        b = np.zeros(size)

        # Compute second-order spatial derivatives
        Q_xx = np.zeros_like(Q)
        Q_xx[1:-1, :] = Q[2:, :] - 2 * Q[1:-1, :] + Q[0:-2, :]

        # Compute first-order spatial derivatives
        F_x = np.zeros_like(Q)
        F_x[1:-1, :] = (F[2:, :] - F[0:-2, :]) / (2 * dx)

        # RHS of beam warming equation
        RHS = -dt * F_x + epsilon_e * Q_xx / dx**2

        # Flatten RHS
        # NOTE: Flatten for easy operations
        b = RHS.flatten()

        # LHS of beam warming equation
        for i in range(nx):
            idx = i * 3
            I = np.eye(3)
            A_i = A_matrices[i] # Define Jacobian matrix at point i
            # LHS block -> This step for block tridiagonal
            LHS_block = I + theta * dt * A_i / dx
            # Implicit artificial dissipation coeffieicnt term in equation 
            LHS_block -= epsilon_i * (np.roll(I, -1, axis=0) - 2 * I + np.roll(I, 1, axis=0)) / dx**2

            # This is for the main diagonal of the matrix
            for m in range(3):
                for n in range(3):
                    data.append(LHS_block[m, n])
                    rows.append(idx + m)
                    cols.append(idx + n)

            if i > 0:
                # Lower diagonal
                L_i = -theta * dt * A_matrices[i - 1] / (2 * dx)
                for m in range(3):
                    for n in range(3):
                        data.append(L_i[m, n])
                        rows.append(idx + m)
                        cols.append(idx - 3 + n)

            if i < nx - 1:
                # Upper diagonal
                U_i = theta * dt * A_matrices[i + 1] / (2 * dx)
                for m in range(3):
                    for n in range(3):
                        data.append(U_i[m, n])
                        rows.append(idx + m)
                        cols.append(idx + 3 + n)

        # Convert to sparse matrix
        # TODO: Not sure why we need to do this, the program bugs without this step :(
        #       Error regarding the shape of the matrix in the proceesing steps.
        #       FOUNF THE ANSWER -> TO BE ABLE TO USE THE SPARSE SCIPY SOLVER.
        #           IN THE COMPUTE STEP
        A = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

        return A, b

    def computeTimeStep(self):
        Q = self.Q.copy()
        Q = self.applyBoundaryConditions(Q)
        A, b = self.beamWarmingStep(Q)

        try:
            # NOTE: Used a direct solver available in scipy inverse and solve the system.
            delta_Q_flat = splinalg.spsolve(A, b)
            delta_Q = delta_Q_flat.reshape(self.nx, 3)
            # Update conserved variables
            self.Q += delta_Q
            self.Q = self.applyBoundaryConditions(self.Q)
        except splinalg.MatrixRankWarning:
            print("Matrix is singular at this time step.")

        self.Q[:, 0] = np.maximum(self.Q[:, 0], 1e-6)
        p = self.computePressure(self.Q)
        self.Q[:, 2] = self.Q[:, 2].clip(min=0.0)

    def computeBeamWarming(self):
        time_step = self.nt
        for n in range(time_step):
            self.computeTimeStep()
            if n % 100 == 0:
                self.logger.info(f"Time step: {n}/{time_step:>10.4f}")

    def plotResults(self, showExact=False, showInitial=True):
        rho = self.Q[:, 0]
        u = self.Q[:, 1] / self.Q[:, 0]
        p = self.computePressure(self.Q)
        
        x = np.linspace(0, self.nx * self.dx, self.nx)
        initialLinewidth = 1.0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        if showInitial:
            ax1.plot([0, 500, 500, 1000], [1, 1, 4, 4], 'r--', label='Initial', linewidth=initialLinewidth)
        ax1.plot(x, rho, 'b-', label='t = 250')

        ax1.set_ylabel(r'$\rho$')
        ax1.set_xlabel(r'$x$')
        ax1.grid(True)
        ax1.legend()
        
        if showInitial:
            ax2.plot([0, 1000], [0, 0], 'r--', label='Initial', linewidth=initialLinewidth)
        ax2.plot(x, u, 'b-', label='t = 250')

        ax2.set_ylabel(r'$u$')
        ax2.set_xlabel(r'$x$')
        ax2.grid(True)
        ax2.legend()
        
        if showInitial:
            ax3.plot([0, 500, 500, 1000], [1, 1, 4, 4], 'r--', label='Initial', linewidth=initialLinewidth)
        if showExact:
            ax3.plot(df.x, df.p, 'g-', label='Exact', linewidth=0.75)
        ax3.plot(x, p, 'b-', label='t = 250')

        ax3.set_ylabel(r'$p$')
        ax3.set_xlabel(r'$x$')
        ax3.grid(True)
        ax3.legend()
        plt.tight_layout()
        plt.savefig('figs/beamwarming_q1_appendix.eps', format='eps')
        plt.show()

    def updatePlotSettings(self):
        plt.rcParams.update({
            'font.family': 'serif',  
            'font.serif': ['Times New Roman'],  
            'font.size':       11,  
            'axes.titlesize':  11,  
            'axes.labelsize':  11,  
            'legend.fontsize': 9, 
            'xtick.labelsize': 11, 
            'ytick.labelsize': 11  
        })  

    def plotPressureOnly(self):
        p = self.computePressure(self.Q)
        x = np.linspace(0, self.nx * self.dx, self.nx)
        plt.figure(figsize=(8,4))
        plt.plot(df.x, df['p'], 'g-', label='Theory', linewidth=1.0)
        plt.plot(x, p, 'b-', label=f'Implicit CFL = {self.cfl} $\epsilon_e$ = {self.epsilon_e}')
        plt.ylabel(r'$P/P_r$')
        plt.xlabel(r'$X$')
        plt.legend()
        plt.gca().spines['right'].set_visible(False) 
        plt.gca().spines['top'].set_visible(False)
        plt.savefig('figs/beamwarming_q1.eps', format='eps')
        plt.show()