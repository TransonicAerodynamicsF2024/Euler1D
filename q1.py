from modules.shockTube_MacCormack import shockTube_MacCormack
from modules.shockTube_BeamWarming import shockTube_BeamWarming

def macCormack():
    """
    NOTE
    This functions initializes a shock tube object and 
    computes the solution using the MacCormack method.
    """
    macC = shockTube_MacCormack()
    macC.setInitialConditions()
    macC.computeMacCormack(250.0, 0.1)
    macC.plotResults(showInitial=True, showExact=True)
    macC.plotPressureOnly()

def beamWarming():
    """
    NOTE
    This functions initializes a shock tube object and 
    computes the solution using the implicit Beam Warming Method.
    """
    beamW = shockTube_BeamWarming(length=1000.0, nx=1000, t_max=250.0, cfl=0.5)
    beamW.setInitialConditions()
    beamW.computeBeamWarming()
    beamW.plotResults(showExact=True, showInitial=True)
    beamW.plotPressureOnly()


if __name__ == "__main__":
    macCormack()
    beamWarming()