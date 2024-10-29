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

df = pd.read_csv('modules/exact.csv')
df['p'] = df['p'].values[::-1]

@dataclass
class FluidProperties:
    gamma: float = 1.4

class shockTube_MacCormack:
    def __init__(self, nx:int=1000, 
                 length:float=1000.0, 
                 gamma:float=1.4) -> None:
        
        self.nx = nx
        self.dx = length / nx
        self.gamma = gamma
        self.fluid = FluidProperties(gamma=gamma)

        # Initialize the arrays for conservative variables
        self.rho = np.zeros(nx)
        self.rhou = np.zeros(nx)
        self.e = np.zeros(nx) # energy
        self.u = np.zeros(nx)
        self.p = np.zeros(nx)

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
    
    def macCormackStepSod(self, dt):
 
        Q = np.array([self.rho, self.rhou, self.e])
        Q_pred = np.zeros_like(Q)
        Q_new = np.zeros_like(Q)
        
        # Predictor step (forward difference)
        F = np.array([self.computeFlux(self.rho[i], self.rhou[i], self.e[i]) for i in range(self.nx)]).T

        for i in range(1, self.nx-1):
            Q_pred[:, i] = Q[:, i] - dt/self.dx * (F[:, i+1] - F[:, i])
        
        # Zero order extrapolation after the predictor step.
        Q_pred[:, 0] = Q_pred[:, 1]
        Q_pred[:, -1] = Q_pred[:, -2]
        
        # Compute fluxes for predicted values
        F_pred = np.array([self.computeFlux(Q_pred[0, i], Q_pred[1, i], Q_pred[2, i]) for i in range(self.nx)]).T
        
        # Corrector step (backward difference)
        for i in range(1, self.nx-1):
            Q_new[:, i] = 0.5 * (Q[:, i] + Q_pred[:, i] - dt/self.dx * (F_pred[:, i] - F_pred[:, i-1]))
        
        # Apply boundary conditions
        Q_new[:, 0] = Q_new[:, 1]
        Q_new[:, -1] = Q_new[:, -2]
        
        # Update solution
        self.rho = Q_new[0]
        self.rhou = Q_new[1]
        self.e = Q_new[2]
        
        # Update primitive variables
        self.u = self.rhou / self.rho
        self.p = (self.gamma - 1.0) * (self.e - 0.5 * self.rho * self.u**2)

    def computeMacCormack(self, t_final, dt):
        n_steps = int(t_final / dt)

        for step in range(n_steps):
            if step == 0:
                self.logger.info(f"{'Step':>10} {'Time':>10}")
            
            self.macCormackStepSod(dt)
            
            # Log results
            if (step + 1) % 100 == 0:
                self.logger.info(f"{step+1:>10} {step*dt:>10.2f}{'x':>10} {'rho':>10} {'u':>10} {'p':>10}")
                for i in range(0, self.nx, 200):
                    self.logger.info(f"{'':>20} {i*self.dx:>10.2f} {self.rho[i]:>10.2f} {self.u[i]:>10.2f} {self.p[i]:>10.2f}")


    def plotResults(self, showInitial=True, showExact=False):
        x = np.linspace(0, self.nx * self.dx, self.nx)
        initialLinewidth = 1.0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        # Plot density
        if showInitial:
            ax1.plot([0, 500, 500, 1000], [1, 1, 4, 4], 'r--', label='Initial', linewidth=initialLinewidth)
        ax1.plot(x, self.rho, 'b-', label='t = 250')
        ax1.set_ylabel(r'$\rho$')
        ax1.set_xlabel(r'$x$')
        ax1.grid(True)
        ax1.legend()
        
        # Plot velocity
        if showInitial:
            ax2.plot([0, 1000], [0, 0], 'r--', label='Initial', linewidth=initialLinewidth)
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
        plt.savefig('figs/macCormack_q1_appendix.eps', format='eps')
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
        x = np.linspace(0, self.nx * self.dx, self.nx)
        plt.figure(figsize=(8,4))
        plt.plot(df.x, df['p'], 'g-', label='Theory', linewidth=1.0)
        plt.plot(x, self.p, 'b-', label=f'MacCormack CFL = 0.5')
        plt.ylabel(r'$P/P_r$')
        plt.xlabel(r'$X$')
        plt.legend()
        plt.gca().spines['right'].set_visible(False) 
        plt.gca().spines['top'].set_visible(False)
        plt.savefig('figs/macCormack_q1.eps', format='eps')
        plt.show()