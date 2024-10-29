from modules.nozzle_MacCormack import nozzle_MacCormack
from modules.nozzle_BeamWarming import nozzle_BeamWarming

def macCormack_nozzle():
    """
    NOTE
    This functions initializes a shock tube object and 
    computes the solution using the MacCormack method.
    """
    macC = nozzle_MacCormack(nx=100, cfl=0.5, outflow='subsonic')
    macC.setInitialConditions()
    macC.computeMacCormack(maxIters=25000)
    macC.plotResiduals()
    macC.plotResults()

def beamWarming_nozzle():
    """
    NOTE
    This functions initializes a shock tube object and 
    computes the solution using the Beam-Warming method.
    """
    beamW = nozzle_BeamWarming(nx=100, cfl=0.5, outflow='supersonic')
    beamW.setInitialConditions()
    beamW.computeBeamWarmingNew(maxIters=25000)
    beamW.plotResiduals()
    beamW.plotResults()
    
if __name__ == "__main__":
    macCormack_nozzle()
    beamWarming_nozzle()