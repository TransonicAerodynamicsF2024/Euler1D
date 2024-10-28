from modules.nozzle_MacCormack import nozzle_MacCormack
from modules.nozzle_BeamWarming import nozzle_BeamWarming

def macCormack_nozzle():
    """
    This functions initializes a shock tube object and 
    computes the solution using the MacCormack method.
    """
    macC = nozzle_MacCormack(nx=100)
    macC.setInitialConditions(showNozzle=True)
    macC.computeMacCormack(maxIters=15000, cfl=0.5)
    macC.plotResiduals()
    macC.plotResults()


if __name__ == "__main__":
    macCormack_nozzle()
