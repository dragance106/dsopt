The code implements a Monte Carlo-based method for parallel surrogate single-objective optimization of black-box functions defined on discrete spaces. 
The main method is in the file dsopt.py, while tests.py contains the code for comparing its efficiency against 
another Monte Carlo-based method SOP from pySOT package, 
Bayesian optimization and random forests variants from smac3 package,
and mixed variable genetic algorithm from pymoo package.

The new method is described in more detail in the forthcoming paper
H. Dashti, S. Stevanović, S. Al-Yakoob, D. Stevanović,
*Parallel Monte Carlo surrogate optimization of semi-expensive black-box functions in discrete spaces*,
accepted for publication in Journal of Building Engineering (2025).

Work on this project was supported by Kuwait University through small project VA01/23.
