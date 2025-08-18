
# dsopt

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

dsopt is an open-source Python package 
implementing a practical, parallel surrogate-optimization workflow 
for expensive black-box functions over mixed-type design spaces. 
It treats continuous, integer and categorical variables uniformly 
by representing each parameter as a finite set of admissible values. 
dsopt combines fast XGBoost surrogates, lightweight uncertainty metrics, a Monte-Carlo candidate pool 
and a sampling strategy that mixes exploitative, Pareto and explorative points 
to form parallel evaluation batches without costly inner acquisition solvers. 
With predictable runtimes and minimal dependencies, 
dsopt makes surrogate-assisted optimization accessible 
to researchers and practitioners with basic Python skills.

---

## 📦 Installation

### Requirements
- Python >= 3.8  
- Dependencies
  - `pandas`
  - `numpy`  
  - `xgboost`  
  - `scikit-learn`

### Install from Source (GitHub)
```bash
git clone https://github.com/dragance106/dsopt.git
cd dsopt
pip install .
```
You can also download the pure Python wheel from [GitHub](https://github.com/dragance106/dsopt/releases/tag/v1.0.0)
and install it with `pip`.

---

## 🚀 Usage

### Minimal working example
```python
from dsopt import dsopt
import numpy as np

# Define a simple black-box function [Gramacy & Lee, 2012](https://www.sfu.ca/~ssurjano/grlee12.html)
# This function is usually evaluated on x ∈ [0.5, 2.5]
# Minimum around -0.9 for x around 0.55
def grlee12(x):
    term1 = np.sin(10*np.pi*x) / (2*x)
    term2 = (x-1)**4
    return term1 + term2

# Run optimization
best_value, best_point = dsopt(grlee12,
                               {'x': np.linspace(0.5, 2.5, 2001)},
                               opt='min',
                               max_evaluations=100,
                               initial_sample_size=12,
                               iterative_sample_size=8)

print("Best objective value: ", best_value)
print("Best point: ", best_point)
```

### 📊 Sample Output
```
Best objective value:  -0.8689254470111416
Best point:  {'x': np.float64(0.549)}
```

In the above code, the first argument of `dsopt` is a Python callable
that represents the expensive function to be optimized.
The second argument is a dictionary describing the design space for the expensive function.
In this case, the *expensive* function `grlee12` has a single real parameter `x`
that may take values from the interval [0.5, 2.5]. 
For `dsopt` real parameters should be discretized:
here it is done by calling `np.linspace(0.5, 2.5, 2001)`
which returns a list of uniformly spaced values 0.5, 0.501, 0.502, ..., 2.499, 2.5.

The remaining arguments mean that:
- the expensive function should be minimized (`opt=min`)
- through at most 100 evaluations (`max_evaluations=100`), and that
- `dsopt` should start with a random sample of 12 points (`initial_sample_size=12`)
- after which in each new iteration 8 new points will be sampled for further evaluations (`iterative_sample_size=8`).

Methodology behind `dsopt` is described in a recent publication:
[H Dashti, S Stevanović, S Al-Yakoob, D Stevanović, 
Parallel Monte Carlo-based surrogate optimization of building energy models, 
J Build Eng 107 (2025) 112579](https://doi.org/10.1016/j.jobe.2025.112579).

### A more useful example
```python
from dsopt.dsopt import dsopt
from kw_villa.kw_villa import simulate_desert_villa
import numpy as np

best_value, best_point = dsopt(simulate_desert_villa,
                               {'glazing_open_facade': ([1, 2, 3, 4, 5, 6], 'c'),
                                'shading_open_facade': (['int_shade', 'ext_shade', 'ext_blind'], 'c'),
                                'glazing_closed_facade': ([1, 2, 3, 4, 5, 6], 'c'),
                                'wwr_front': np.linspace(0.05, 0.9, 18),
                                'exterior_wall': ([1, 2, 3, 4], 'c'),
                                'insulation_thickness': np.linspace(0.0, 0.5, 11)},
                               opt='min',
                               initial_sample_size=32,
                               iterative_sample_size=8,
                               max_evaluations=400,
                               k=100,
                               uncertainty_metric='mipt',
                               omega_percentage=10,
                               tau_percentage=10,
                               upsilon_percentage=25,
                               sigma_t=50,
                               sigma_u=10,
                               gamma=0.01,
                               r=20,
                               verbose_level=1,
                               filename=f'dsopt_mipt_kw_villa.csv')

print("Minimum building energy ", best_value)
print("attained for design space parameters: ", best_point)
```

### 📊 Sample Output
```
Minimum building energy  760504461436.1876
attained for design space parameters: {'glazing_open_facade': 3, 'shading_open_facade': 'ext_shade', 'glazing_closed_facade': 6, 'wwr_front': 0.39999999999999997, 'exterior_wall': 2, 'insulation_thickness': 0.5}
```

In this case `simulate_desert_villa` is a method from `tests\kw_villa\kw_villa.py`
that uses [`eppy`](https://eppy.readthedocs.io/en/latest/) 
to set up a programmable EnergyPlus model of a residential villa in Kuwait,
call EnergyPlus to simulate its energy behavior and
returns the total primary energy needed for heating and cooling (well, mostly cooling).

The design space parameters now include categorical variables
`glazing_open_facade`, `shading_open_facade`, `glazing_closed_facade` and `exterior_wall`,
indicated as a tuple containing the list of admissible values and the categorical indicator `'c'`.
`wwr_front` and `insulation_thickness` are real parameters
discretized with a relatively small resolution,
indicative of expected resolution with which
they might actually be used in a constructed building
(one could hardly expect insulation thickness of 0.324875m, for example).

One should note that `dsopt` can also handle arrays of related design space parameters:
for example, `'x': (10, ['left', 'right'], 'c')`
would indicate an array of ten categorical parameters `'x0', ... 'x9'`
each having a value either `'left'` or `'right'`.

The remaining arguments in the above call give a greater control over the inner workings of `dsopt`.
At moment, three different uncertainty metrics are supported:
`'mipt'` and `'voronoi'` are distance based,
while `'aposteriori'` uses historical relative errors of earlier instances of the surrogate model.

`dsopt` selects a new sample by taking a large pool of candidates 
(whose size is controlled by the value of the multiplier `k`),
which is divided into areas of low uncertainty, medium uncertainty and high uncertainty.
`tau_percentage` sets the percentage of the candidate pool that is proclaimed as low uncertainty,
where it is expected that the surrogate model predicts the expensive function relatively well,
so that only the candidates with predictions close to the best one observed so far
(lagging by at most `omega_percentage` percents behind it) are kept. 
Among these, the candidates with the best predictions (*exploitative* points) 
will form `sigma_t` percents of the new evaluation sample.

On the other hand, 
`upsilon_percentage` sets the percentage of the candidate pool that is proclaimed as high uncertainty,
where it is expected that the surrogate model predictions will deviate significantly from the expensive function.
In this region we select only the candidates with highest uncertainty (*explorative* points)
which will form `sigma_u` percents of the new evaluation sample.

All the remaining candidates form the area of medium uncertainty,
where it is expected that the surrogate model prediction quality will decrease with increasing uncertainty.
Hence, higher uncertainty implies that one should primarily be interested in candidates with better predictions,
so that in this area `dsopt` finds Pareto solutions for with increasing (prediction, uncertainty) values
and samples uniformly among them to create
the remaining `100-sigma_t-sigma_u` percents of the new evaluation sample.

Several more arguments exist for `dsopt`,
whose description can be found in its docstring in `dsopt/dsopt.py`.

### A few further examples

`tests` folder contains further examples of using `dsopt`
with categorical versions of the COCO mixed-integer test functions,
and their benchmarking against [pysot](https://github.com/dme65/pySOT), [pymoo](https://pymoo.org/) and [smac3](https://github.com/automl/SMAC3).

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Coding Standards
- Follow **PEP8** coding style.
- Add **docstrings** to all public functions and classes.
- Include **tests** for any new functionality.

### Workflow
1. Fork the repository.  
2. Create a new branch for your feature:  
   ```bash
   git checkout -b feature/my-feature
   ```
3. Commit changes with clear messages.  
4. Push to your fork and submit a Pull Request (PR).  

### Branch Management
- `main` – stable, production-ready code.  
- `dev` – active development branch.  
- Feature branches – for experimental or incremental changes.  

---

## 📄 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact & Support

- **Authors:** Sanja Stevanović and Dragan Stevanović  
- **Email:** sanja.stevanovic@mi.sanu.ac.rs, dragance106@yahoo.com  
- **Issues & Discussions:** [GitHub Issues](https://github.com/dragance106/dsopt/issues)  

If you encounter bugs, have questions, or want to suggest features, 
please open an issue or start a discussion on GitHub.  

---
