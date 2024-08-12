# Barn Conditional Sampling Problem

## What is it about?
- Picking a section of the barn in Z-axis, find k points in which the average of the CO2 concentration at these points equals to the average of CO2 concentration of the entire barn.

## Install requirements
- Install PyTorch at: https://pytorch.org/
- Install other requirements by: `pip install -r requirements.txt`

## How to use it?
- Run the script, then the corresponding results will appear in the results folder:

`
python barnCSP.py -c [tda-mapper/kmedoids/random/uniform/simulated-annealing/PSO/monte-carlo] -d [2D/3D] [path/to/barn/csv/file]
`

### Note
Extra settings are configured directly in the configuration dictionary in the barnCSP.py file