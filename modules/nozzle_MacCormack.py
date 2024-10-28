import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass

import sys
np.set_printoptions(threshold=sys.maxsize)

@dataclass
class FluidProperties:
    gamma: float = 1.4

class nozzle_MacCormack:
    def __init__(self, nx:int=10,
                 length:float=10.0, 
                 gamma:float=1.4, 
                 mach_inlet:float=1.30, 
                 outflow:str='supersonic') -> None:
        
        self.nx = nx
        self.dx = length / nx
        self.gamma = gamma
        self.mach_inlet = mach_inlet
        self.outflow = outflow
        self.fluid = FluidProperties(gamma=gamma)

        
        # Initialize variables
        self.rho = np.ones(nx)
        self.rhou = np.ones(nx)
        self.e = np.ones(nx)
        self.u = np.ones(nx)
        self.p = np.ones(nx)
        
        # Area and area derivative
        self.A = 1.398 + 0.347 * np.tanh(0.8 * np.linspace(0, length, nx) - 4)
        self.dA_dx = np.gradient(self.A, self.dx)

        self.setupLogging()

    def setupLogging(self):
        """
        TODO - change the logging function for beam warming method
        configure the logging to write results to console
        """
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def setInitialConditions(self, showNozzle=False):
        """
        NOTE: Reference from the textbook:

        The nozzle is of length 10, with the incoming flow supersonic (M = 1.3) and
        the outgoing flow subsonic (M < 1). The entire flowfield is initialized with the
        incoming flow values.
        """
        self.rho[:] = 1.0 - 0.05 * np.linspace(0, 10, self.nx)
        self.p[:] = 0.09 * np.linspace(0, 10, self.nx) + 1.0
        self.u[:] = self.mach_inlet * np.sqrt(self.gamma * self.p / self.rho)
        self.rhou[:] = self.rho * self.u
        self.e[:] = self.p / (self.gamma - 1) + 0.5 * self.rho * self.u**2

        if showNozzle:
            x = np.linspace(0, self.nx * self.dx, self.nx)
            linew = 4.0
            
            plt.figure(figsize=(10, 4))
            plt.plot(x, 0.5 * self.A, 'k-', linewidth=linew)
            plt.plot(x, -0.5 * self.A, 'k-', linewidth=linew)
            plt.gca().axes.yaxis.set_visible(False) 
            plt.gca().spines['left'].set_visible(False) 
            plt.gca().spines['right'].set_visible(False) 
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks([])
            plt.title('Nozzle Geometry')
            plt.show()

    def computeFlux(self, rho, rhou, e):
        u = rhou / rho
        p = (self.gamma - 1.0) * (e - 0.5 * rho * u**2)
        
        f1 = rhou * self.A
        f2 = (rhou * u + p) * self.A
        f3 = self.A * (e + p) * u
        return np.array([f1, f2, f3])

    def computePrimitives(self, rho, rhou, e):
        u = np.zeros_like(rho)
        p = np.zeros_like(rho)
        u[1:-1] = rhou[1:-1] / rho[1:-1]
        p[1:-1] = (self.gamma - 1) * (e[1:-1] - 0.5 * rho[1:-1] * u[1:-1]**2)
        return u, p
    
    def applyBoundaryConditions(self, outflow='supersonic'):
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

        self.p[0] = 1.0
        self.rho[0] = 1.0
        self.u[0] = self.mach_inlet * np.sqrt(self.gamma * self.p[0] / self.rho[0])
        self.rhou[0] = self.rho[0] * self.u[0]
        self.e[0] = self.p[0] / (self.gamma - 1) + 0.5 * self.rho[0] * self.u[0]**2

        if outflow == 'supersonic':
            """
            NOTE: Extrapolation is used for the supersonic case.
            """
            self.rho[-1] = self.rho[-2]
            self.u[-1] = self.u[-2]
            self.p[-1] = self.p[-2]
            self.rhou[-1] = self.rho[-1] * self.u[-1]
            self.e[-1] = self.p[-1] / (self.gamma - 1) + 0.5 * self.rho[-1] * self.u[-1]**2

        elif outflow == 'subsonic':
            """
            NOTE: Riemann invariants are used for the subsonic case.
            Additionally, the back pressure is set to 1.9 * p_inlet.
            """
            # Specify exit pressure
            self.p[-1] = 1.9 * self.p[0]
            c_Nm1 = np.sqrt(self.gamma * self.p[-2] / self.rho[-2])

            # Compute Riemann invariant from interior point
            R1 = self.u[-2] + (2 * c_Nm1) / (self.gamma - 1)

            # Compute density at outlet using isentropic relation
            self.rho[-1] = self.rho[-2] * (self.p[-1] / self.p[-2])**(1 / self.gamma)

            # Compute speed of sound at outlet
            c_N = np.sqrt(self.gamma * self.p[-1] / self.rho[-1])

            # Compute velocity at outlet using R1
            self.u[-1] = R1 - (2 * c_N) / (self.gamma - 1)

            # Update conservative variables
            self.rhou[-1] = self.rho[-1] * self.u[-1]
            self.e[-1] = self.p[-1] / (self.gamma - 1) + 0.5 * self.rho[-1] * self.u[-1]**2

        else:
            raise ValueError("Invalid outflow condition. Choose 'supersonic' or 'subsonic'")

    def computeDt(self, u, c, dx, CFL_max) -> float:
        """
        NOTE: Reference from the textbook:
        Choose the minimum timestep as shown in the equation below:

        CFL = (|u| + c) * dt/dx

        This condition must be satisfied at each computational point, and the 
        timestep must be taken as the smallest one out of the computational 
        domain.

        Returns:
        --------
        dt      : float
            Time step satisfying the CFL condition.
        """

        # Compute the local wave speed at each grid point
        local_wave_speed = np.abs(u) + c

        # Avoid division by zero in case local_wave_speed has zeros
        # Replace zeros with a very small number (machine epsilon)
        local_wave_speed = np.where(local_wave_speed == 0, np.finfo(float).eps, local_wave_speed)

        # Compute dt at each grid point
        dt_local = CFL_max * dx / local_wave_speed

        # Take the minimum dt over the computational domain
        dt = np.min(dt_local)

        return dt

    def macCormackStep(self, dt):
        # Conservative variables

        Q = np.array([self.rho * self.A, self.rhou * self.A, self.e * self.A], dtype=float)
        Q_pred = np.zeros_like(Q)
        Q_new = np.zeros_like(Q)

        # Compute flux and source term
        F = self.computeFlux(self.rho, self.rhou, self.e)
        S = np.zeros((3, self.nx))
        S[1, :] = self.p * self.dA_dx  # Only second component has non-zero source term

        # Predictor step
        for i in range(1, self.nx - 1):
            Q_pred[:, i] = (Q[:, i]) - (dt / self.dx) * (F[:, i + 1] - F[:, i]) + (dt * S[:, i])

        self.rho = Q_pred[0] / self.A
        self.rhou = Q_pred[1] / self.A
        self.e = Q_pred[2] / self.A

        self.u, self.p = self.computePrimitives(self.rho, self.rhou, self.e)
        self.applyBoundaryConditions(outflow=self.outflow)

        # Compute fluxes for predicted values
        F_pred = self.computeFlux(self.rho, self.rhou, self.e)

        # Corrector step
        for i in range(1, self.nx - 1):
            Q_new[:, i] = 0.5 * (Q[:, i] + Q_pred[:, i] - dt / self.dx * (F_pred[:, i] - F_pred[:, i - 1]) + dt * S[:, i])

        # Update solution with boundary conditions and enforce positive bounds for stability
        self.rho = Q_new[0] / self.A
        self.rhou = Q_new[1] / self.A
        self.e = Q_new[2] / self.A
        self.u, self.p = self.computePrimitives(self.rho, self.rhou, self.e)
        self.applyBoundaryConditions(outflow=self.outflow)     

   
    def computeMacCormack(self, maxIters = 10000, cfl=0.5):
        dt = self.computeDt(self.u, np.sqrt(self.gamma * self.p / self.rho), self.dx, cfl)
        self.residuals = []
        ss = 0
        while ss < maxIters:
            rho_old = np.copy(self.rho)
            self.macCormackStep(dt)
            residual = np.sqrt(np.sum((self.rho - rho_old)**2))
            self.residuals.append(residual)

            ss += 1

            if ss % 100 == 0:
                print(f"Step: {ss}, dt: {dt:.5e}, Residual: {residual:.5e}, Time: {ss * dt:.2f}")
            if residual < 1e-10:
                break
            
    def plotResults(self):

        # plot the initial conditions
        x = np.linspace(0, self.nx * self.dx, self.nx)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
        ax1.plot(x, self.rho, 'r-', label=r'$\rho$')
        ax1.set_ylabel(r'$\rho$')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(x, self.u, 'r-', label=r'$u$')
        ax2.set_ylabel(r'$u$')
        ax2.grid(True)
        ax2.legend()

        ax3.plot(x, self.p, 'r-', label=r'$p$')
        ax3.set_ylabel(r'$p$')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def plotResiduals(self):
        x = np.linspace(0, self.nx * self.dx, self.nx)
        # Compute Mach number from velocity and sound speed
        mach_number = self.u / np.sqrt(self.gamma * self.p / self.rho)
        # Plot Mach number vs x
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Mach number plot
        ax1.plot(x, mach_number, label='Mach Number')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Mach Number')
        ax1.grid(True)
        ax1.legend()

        # Residuals plot
        ax2.plot(self.residuals, label='Residuals')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Residual')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()
