# SodShockTube-and-Quasi1DNozzle
Programmer: ```Paramvir Lobana```

## Usage
The modules directory contain all the modules for the homework question. Referring to 
the directory structure section, the modules are named according to their respective functions.
The modules are called in the main python files: ```q1.py``` and ```q2.py```. \

Running these files will run the particular problem using both the methods.

### Example Usage
The following code is an example of how to define a shock tube or a nozzle object and call the solver solve the problem.
The modules can be imported as follows:

```python
from modules.shockTube_MacCormack import shockTube_MacCormack
from modules.shockTube_BeamWarming import shockTube_BeamWarming
from modules.nozzle_MacCormack import nozzle_MacCormack
from modules.nozzle_BeamWarming import nozzle_BeamWarming
```
The example below shows how to call the solver for the nozzle problem using the MacCormack method.
The procedure is similar for all the procedures:

```python
macC = nozzle_MacCormack()
macC.setInitialConditions(showNozzle=True)
macC.computeMacCormack(maxIters=15000, cfl=0.5)
macC.plotResiduals()
macC.plotResults()
```
The first line defines an object for the nozzle problem. This initiates all the default variables in the background.
Then the initial conditions are set. The *showNozzle* flag displays the nozzle shape and area. It is set to ```False``` by default.
Then, in the third line, the solver is called and solves the problem using the provided initial conditons.
The residuals and results can be plotted using the ```plotResiduals``` and ```plotResults``` methods respectively.

## Directory Structure
```bash
SodShockTube-and-Quasi1DNozzle
├── README.md
├── modules
│   ├── __init__.py
│   ├── exact.csv
│   ├── nozzle_BeamWarming.py
│   ├── nozzle_MacCormack.py
│   ├── shockTube_BeamWarming.py
│   └── shockTube_MacCormack.py
├── q1.py
└── q2.py
```


