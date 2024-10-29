#   Purpose:
#       - This script contains the class definition for the Nozzle class.
#
#   Record of Revision:
#       Date            Programmer              Description
#       ====            ==========              ===========
#       21 Oct, 2024    Paramvir Lobana         Original Code

import numpy as np
import logging
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class FluidProperties:
    gamma: float = 1.4

class nozzle_BeamWarming:
    def __init__(self, 
                 length=10.0, 
                 nx=201, 
                 t_max=500.0, 
                 gamma=1.4, 
                 mach_inlet=1.25,
                 cfl = 0.25,
                 outflow='subsonic') -> None:

        self.length = length        
        self.nx = nx                
        self.dx = length / (nx - 1) 
        self.t_max = t_max          
        self.gamma = gamma          
        self.mach_inlet = mach_inlet
        self.cfl = cfl
        self.outflow = outflow

        self.dt = 0.005
        self.nt_max = int(self.t_max / self.dt)

        self.epsilon_e = 0.1
        self.epsilon_i = 2.5 * self.epsilon_e
        self.theta = 1.0

        self.x = np.linspace(0, self.length, self.nx)
        self.A = 1.398 + 0.347 * np.tanh(0.8 * np.linspace(0, length, nx) - 4)
        self.dA_dx = np.gradient(self.A, self.dx)

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
        p0 = 1.0
        rho0 = 1.0
        c0 = np.sqrt(self.gamma * p0 / rho0)
        u0 = self.mach_inlet * c0
        E0 = p0 / (self.gamma - 1) + 0.5 * rho0 * u0 ** 2

        # Initialize Q 
        for i in range(self.nx):
            self.Q[i, 0] = rho0 * self.A[i]
            self.Q[i, 1] = rho0 * u0 * self.A[i]
            self.Q[i, 2] = E0 * self.A[i]

    def computePressure(self, Q):
        rho = Q[:, 0] / self.A
        rho_u = Q[:, 1] / self.A
        e = Q[:, 2] / self.A
        u = rho_u / rho
        p = (self.gamma - 1) * (e - 0.5 * rho * u ** 2)
        return p

    def computeFlux(self, Q):

        rho = Q[:, 0] / self.A
        rho_u = Q[:, 1] / self.A
        e = Q[:, 2] / self.A
        u = rho_u / rho
        p = self.computePressure(Q)
        F = np.zeros_like(Q)
        F[:, 0] = rho_u * self.A
        F[:, 1] = (rho_u * u + p) * self.A
        F[:, 2] = u * (e + p) * self.A
        return F

    def computeSourceTerm(self, Q):
        p = self.computePressure(Q)
        S = np.zeros_like(Q)
        S[:, 1] = p * self.dA_dx
        return S

    def computeJacobian(self, Q):
        """
        NOTE: Reference from textbook
        The Jacobian matrix A is constructed at all points for the implicit
        steps and, since it operates on a 1st-derivative central difference operator,
        it is computed for the lower and the upper band of the block diagonal matrix.
        """
        rho = Q[:, 0] / self.A
        rho_u = Q[:, 1] / self.A
        e = Q[:, 2] / self.A
        u = rho_u / rho
        p = self.computePressure(Q)
        H = (e + p) / rho

        # TODO - Change variable to A_jac since A is the variable for area
        #        in this case

        A_jac = np.zeros((self.nx, 3, 3))
        for i in range(self.nx):
            A_i = np.zeros((3, 3))
            A_i[0, 0] = 0.0
            A_i[0, 1] = 1.0
            A_i[0, 2] = 0.0

            A_i[1, 0] = (self.gamma - 3) * (u[i] ** 2) / 2
            A_i[1, 1] = (3 - self.gamma) * u[i]
            A_i[1, 2] = self.gamma - 1

            A_i[2, 0] = self.gamma * u[i] * H[i] - (self.gamma - 1) * u[i] ** 3
            A_i[2, 1] = self.gamma * H[i] - 1.5 * (self.gamma - 1) * u[i] ** 2
            A_i[2, 2] = self.gamma * u[i]
            A_jac[i] = A_i
        return A_jac

    def applyBoundaryConditions(self, Q):
        """
        NOTE: Refernce from the textbook:

        The incorporation of the source term poses no difficulty but the non-reflecting 
        boundary conditions must now be changed to subsonic Riemman type boundary conditions 
        when subsonic flow is present and to supersonic boundary conditions when supersonic 
        flow is present. The supersonic boundary conditions are (with right running characteristics) 

        rho_1 = rho_inf 
        u_1 = u_inf 
        p_1 = p_inf at the inlet, and 

        rhoN = rho{N-1} 
        u_N = u_{N-1} 
        p_N = p_{N-1} at the outlet. 

        The following equation must be used when subsonic flow is present: 

        along u: s = constant 
        along u + c : u + 2 c / (gamma - 1) = R_1 
        along u + c : u - 2 c / (gamma - 1) = R_2 

        where s is the entropy, u is the flow velocity, c is the speed of sound. 
        Since in our model problem the nozzle entrance will be supersonic and the nozzle exit subsonic, 
        we will develop the appropriate boundary condition for the exit station i = N. 
        The two Riemann invariants meeting at the boundary point J are emanating from the point 
        i = N - 1 and from downstream infinity, where the pressure is give the exit station i = N 
        and the from downstream infinity, where the pressure is given. For the rest, please refer 
        to the images I have provided before regarding the boundary condition. I am providing the 
        solution from the textbook for your reference.

        """
        p0 = 1.0
        rho0 = 1.0
        c0 = np.sqrt(self.gamma * p0 / rho0)
        u0 = self.mach_inlet * c0
        E0 = p0 / (self.gamma - 1) + 0.5 * rho0 * u0 ** 2

        Q[0, 0] = rho0 * self.A[0]
        Q[0, 1] = rho0 * u0 * self.A[0]
        Q[0, 2] = E0 * self.A[0]

        if self.outflow == 'supersonic':
            Q[-1, :] = Q[-2, :]

        elif self.outflow == 'subsonic':
            p_back = 1.9 * p0
            rho_im1 = Q[-2, 0] / self.A[-2]
            u_im1 = Q[-2, 1] / Q[-2, 0]
            p_im1 = self.computePressure(Q)[-2]
            c_im1 = np.sqrt(self.gamma * p_im1 / rho_im1)

            # Riemann invariant R1 from interior point
            R1 = u_im1 + (2 * c_im1) / (self.gamma - 1)

            # density at outlet using isentropic relation
            rho_N = rho_im1 * (p_back / p_im1) ** (1 / self.gamma)

            # speed of sound at outlet
            c_N = np.sqrt(self.gamma * p_back / rho_N)

            # velocity at outlet using R1
            u_N = R1 - (2 * c_N) / (self.gamma - 1)

            # conserved variables at outlet
            Q[-1, 0] = rho_N * self.A[-1]
            Q[-1, 1] = rho_N * u_N * self.A[-1]
            E_N = p_back / (self.gamma - 1) + 0.5 * rho_N * u_N ** 2
            Q[-1, 2] = E_N * self.A[-1]
        else:
            raise ValueError("Invalid outflow condition. Choose 'supersonic' or 'subsonic'")

        return Q

    def computeDt(self, u, c, dx, CFL_max) -> float:
        """
        NOTE: Reference from the textbook:
        Choose the minimum timestep as shown in the equation below:

        CFL = (|u| + c) * dt/dx

        This condition must be satisfied at each computational point, and the 
        timestep must be taken as the smallest one out of the computational 
        domain.
        """
        local_wave_speed = np.abs(u) + c
        local_wave_speed = np.where(local_wave_speed == 0, np.finfo(float).eps, local_wave_speed)
        dt_local = CFL_max * dx / local_wave_speed
        dt = np.min(dt_local)

        return dt

    def beamWarmingStep(self, Q, dt):
        # Assemble the LHS matrix and RHS vector
        nx = self.nx
        dx = self.dx
        dt = self.dt
        theta = self.theta
        epsilon_e = self.epsilon_e
        epsilon_i = self.epsilon_i

        # Compute Jacobian matrices, fluxes, and source terms
        A_matrices = self.computeJacobian(Q)
        F = self.computeFlux(Q)
        S = self.computeSourceTerm(Q)

        # Initialize LHS and RHS
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

        # RHS vector
        RHS = - dt * (F_x / dx - S) + epsilon_e * Q_xx / dx ** 2

        # Flatten RHS
        b = RHS.flatten()

        # LHS matrix assembly
        for i in range(nx):
            idx = i * 3
            # Identity matrix
            I = np.eye(3)
            # Jacobian at current point
            A_i = A_matrices[i]
            # LHS block
            LHS_block = I + theta * dt * A_i / dx
            # implicit diffusion term
            LHS_block += epsilon_i * 2 * I / dx ** 2

            for m in range(3):
                for n in range(3):
                    data.append(LHS_block[m, n])
                    rows.append(idx + m)
                    cols.append(idx + n)

            if i > 0:
                # Lower diagonal block
                idx_lower = idx - 3
                for m in range(3):
                    data.append(-epsilon_i * I[m, m] / dx ** 2)
                    rows.append(idx + m)
                    cols.append(idx_lower + m)
            if i < nx - 1:
                # Upper diagonal block
                idx_upper = idx + 3
                for m in range(3):
                    data.append(-epsilon_i * I[m, m] / dx ** 2)
                    rows.append(idx + m)
                    cols.append(idx_upper + m)

            if i > 0:
                # Lower diagonal block
                L_i = -theta * dt * A_matrices[i - 1] / (2 * dx)
                for m in range(3):
                    for n in range(3):
                        data.append(L_i[m, n])
                        rows.append(idx + m)
                        cols.append(idx - 3 + n)
            if i < nx - 1:
                # Upper diagonal block
                U_i = theta * dt * A_matrices[i + 1] / (2 * dx)
                for m in range(3):
                    for n in range(3):
                        data.append(U_i[m, n])
                        rows.append(idx + m)
                        cols.append(idx + 3 + n)

        A = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
        return A, b

    def computeTimeStep(self, dt):
        Q = self.Q.copy()
        Q = self.applyBoundaryConditions(Q)

        A, b = self.beamWarmingStep(Q, dt)

        try:
            delta_Q_flat = splinalg.spsolve(A, b)
            delta_Q = delta_Q_flat.reshape(self.nx, 3)
            self.Q += delta_Q
            self.Q = self.applyBoundaryConditions(self.Q)
        except Exception as e:
            print(f"An error occurred during the linear solve: {e}")

        self.Q[:, 0] = np.maximum(self.Q[:, 0], 1e-6)  
        self.Q[:, 2] = np.maximum(self.Q[:, 2], 1e-6)  

    def computeBeamWarmingNew(self, maxIters=20000):
        ss = 0
        self.residuals = []
        total_time = 0.0
        while ss < maxIters and total_time < self.t_max:

            rho = self.Q[:, 0] / self.A
            rho_u = self.Q[:, 1] / self.A
            u = rho_u / rho
            p = self.computePressure(self.Q)
            c = np.sqrt(self.gamma * p / rho) 
            dt = self.computeDt(u, c, self.dx, self.cfl)

            Q_old = self.Q.copy()

            self.computeTimeStep(dt)

            residual = np.sqrt(np.sum((self.Q - Q_old)**2))
            self.residuals.append(residual)

            ss += 1
            total_time += dt
            # TODO - Too much printing in the console, print at intervals
            # Solution -> Can use the modulo operator to print at intervals
            if ss % 500 == 0:
                self.logger.info(f"Iteration: {ss:>5}, dt: {dt:>8.4e}, Residual: {residual:>10.5e}, Time: {total_time:>5.2f}")

            if residual < 1e-10:
                self.logger.info("Convergence reached.")
                break

    
    def plotResults(self):
        x = np.linspace(0, self.nx * self.dx, self.nx)
        rho = self.Q[:, 0] / self.A
        u = self.Q[:, 1] / self.Q[:, 0]
        p = self.computePressure(self.Q)
        lw = 1.0
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
        ax1.plot(x, rho,  color='black', linewidth=lw)
        ax1.set_ylabel(r'$\rho$')
        ax1.set_xlabel(r'$X$')
        ax1.grid(True)

        ax2.plot(x, u,   color='black', linewidth=lw)
        ax2.set_ylabel(r'$u$')
        ax2.set_xlabel(r'$X$')
        ax2.grid(True)

        ax3.plot(x, p,  color='black', linewidth=lw)
        ax3.set_ylabel(r'$p$')
        ax3.set_xlabel(r'$X$')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('figs/' + self.outflow +'beamwarming_nozzle.eps', format='eps')
        plt.show()

    def save_results(self):
        # TODO
        pass

    def plotResiduals(self):
        rho = self.Q[:, 0] / self.A
        u = self.Q[:, 1] / self.Q[:, 0]
        p = self.computePressure(self.Q)
        x = np.linspace(0, self.nx * self.dx, self.nx)

        lw = 1.0

        mach_number = u / np.sqrt(self.gamma * p / rho)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(x, mach_number, label=f'BeamWarming CFL = {self.cfl}', color='black', linewidth=lw)
        ax1.set_xlabel(r'$X$')
        ax1.set_ylabel('Mach')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(self.residuals, label=f'BeamWarming CFL = {self.cfl}', color='black', linewidth=lw)
        ax2.set_yscale('log')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Log (density error)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('figs/' + self.outflow + 'residuals_beamwarming_nozzle.eps', format='eps')
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