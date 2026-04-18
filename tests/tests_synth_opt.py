# import math
from math import *
import numpy as np

###########################################################################
# Synthetic optimization test functions                                   #
#                                                                         #
# Surjanovic, S. & Bingham, D. (2013).                                    #
# Virtual Library of Simulation Experiments: Test Functions and Datasets. #
# http://www.sfu.ca/~ssurjano/optimization.html                           #
# Retrieved March 13, 2024                                                #
###########################################################################

##########################
# 1-dimensional function #
##########################

# MANY LOCAL MINIMA
# Gramacy & Lee (2012)
# This function is usually evaluated on x ∈ [0.5, 2.5]
# Minimum around -0.9 for x around 0.55
def grlee12(x):
    term1 = np.sin(10*pi*x) / (2*x)
    term2 = (x-1)**4
    return term1 + term2

###########################
# 2-dimensional functions #
###########################

# MANY LOCAL MINIMA
# Holder table function
# The Holder Table function has many local minima, with four global minima.
# The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
# Global minimum -19.2085 at (+-8.05502, +-9.66459)
def holder(x, y):
    fact1 = np.sin(x) * np.cos(y)
    fact2 = np.exp(abs(1 - np.sqrt(x**2 + y**2) / pi))
    return -np.abs(fact1 * fact2)

# MANY LOCAL MINIMA
# Levy N.13 function
# The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
# Global minimum 0 at (1,1)
def levy13(x, y):
    term1 = np.sin(3 * pi * x)**2
    term2 = (x - 1)**2 * (1 + np.sin(3 * pi * y)**2)
    term3 = (y - 1)**2 * (1 + np.sin(2 * pi * y)**2)
    return term1 + term2 + term3

# MANY LOCAL MINIMA
# Shaffer N.2 function
# The function is usually evaluated on the square xi ∈ [-100, 100], for all i = 1, 2.
# Global minimum 0 at (0,0)
def shaffer2(x, y):
    fact1 = np.sin(x**2 - y**2)**2 - 0.5
    fact2 = (1 + 0.001*(x**2 + y**2))**2
    return 0.5 + fact1 / fact2

# MANY LOCAL MINIMA
# Eggholder function
# The Eggholder function is a difficult function to optimize, because of the large number of local minima.
# The function is usually evaluated on the square xi ∈ [-512, 512], for all i = 1, 2.
# Global minimum -959.6407 at (512, 404.2319)
def eggholder(x, y):
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(y + x / 2 + 47)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2

# MANY LOCAL MINIMA
# Drop-wave function
# The function is usually evaluated on the square xi ∈ [-5.12, 5.12], for all i = 1, 2.
# Global minimum -1 at (0,0)
def dropwave(x, y):
    frac1 = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
    frac2 = 0.5 * (x**2 + y**2) + 2
    return -frac1 / frac2

# MANY LOCAL MINIMA
# Cross-in-tray function
# The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
# Global minimum -2.06261 at (+-1.3491, +-1.3491)
# Strange computational behavior here when x1=0 or x2=0
def crossit(x, y):
    fact1 = np.sin(x) * np.sin(y)
    fact2 = np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / pi))
    return -0.0001 * (np.abs(fact1 * fact2) + 1)**0.1

# MANY LOCAL MINIMA
# Bukin N.6 function
# The sixth Bukin function has many local minima, all of which lie in a ridge.
# The function is usually evaluated on the rectangle x1 ∈ [-15, -5], x2 ∈ [-3, 3].
# Global minimum 0 at (-10,1)
def bukin6(x, y):
    term1 = 100 * np.sqrt(np.abs(y - 0.01 * x**2))
    term2 = 0.01 * np.abs(x + 10)
    return term1 + term2

# VALLEY-SHAPED
# Six-hump camel function
# The function has six local minima, two of which are global.
# The function is usually evaluated on the rectangle x1 ∈ [-3, 3], x2 ∈ [-2, 2].
# Global minimum -1.0316 at (0.0898, -0.7126) and (-0.0898, 0.7126)
def camel6(x, y):
    term1 = (4 - 2.1 * x**2 + x**4 / 3) * x**2
    term2 = x * y
    term3 = (-4 + 4 * y**2) * y**2
    return term1 + term2 + term3

# STEEP DROP
# Easom function
# The Easom function has several local minima.
# It is unimodal, and the global minimum has a small area relative to the search space.
# The function is usually evaluated on the square xi ∈ [-100, 100], for all i = 1, 2.
# Global minimum -1 at (pi, pi)
def easom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp( -(x-pi)**2 - (y-pi)**2)

# OTHER
# Beale function
# The Beale function is multimodal, with sharp peaks at the corners of the input domain.
# The function is usually evaluated on the square xi ∈ [-4.5, 4.5], for all i = 1, 2.
# Global minimum 0 at (3, 0.5)
def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

# OTHER
# Branin function
# The Branin function has three global minima.
# This function is usually evaluated on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].
# Global minimum 0.397887 at (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475)
def branin(x, y, a=1, b=5.1/(4*pi**2), c=5/pi, r=6, s=10, t=1/(8*pi)):
    term1 = a * (y - b * x**2 + c * x - r)**2
    term2 = s * (1 - t) * np.cos(x)
    return term1 + term2 + s


###########################
# d-dimensional functions #
###########################

# MANY LOCAL MINIMA
# Ackley on the domain [-32.768, 32.768]^d
# The Ackley function is widely used for testing optimization algorithms.
# Recommended variable values are: a = 20, b = 0.2 and c = 2π.
# The function is usually evaluated on the hypercube xi ∈ [-32.768, 32.768], for all i = 1, …, d,
# although it may also be restricted to a smaller domain.
# Global minimum 0 at (0,0,...,0)
def ackley(xx, a=20, b=0.2, c=2*pi):
    d = len(xx)
    xx=np.array(xx)
    sum1 = np.sum(xx**2)
    sum2 = np.sum(np.cos(c*xx))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# MANY LOCAL MINIMA
# Levy function
# The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, …, d.
# Global minimum 0 at (1,1,...,1)
def levy(xx):
    d = len(xx)
    w = 1 + (np.array(xx)-1)/4
    term1 = np.sin(pi * w[0])**2
    sum1 = np.sum((w-1)**2 * (1+10*np.sin(pi*w + 1)**2)) \
                -(w[d-1]-1)**2 * (1+10*np.sin(pi*w[d-1] + 1)**2)
    term3 = (w[d-1] - 1)**2 * (1 + np.sin(2 * pi * w[d-1])**2)
    return term1 + sum1 + term3

# MANY LOCAL MINIMA
# Rastrigin on the domain [-5.12, 5.12]^dim, dim in {10, 20, 40}
# The Rastrigin function is highly multimodal,
# but locations of the minima are regularly distributed.
# The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
# Global minimum 0 at (0,0,...,0)
def rastrigin(xx):
    d = len(xx)
    xx = np.array(xx)
    sum1 = np.sum(xx**2 - 10*np.cos(2*pi*xx))
    return 10 * d + sum1

# MANY LOCAL MINIMA
# Schwefel function
# The function is usually evaluated on the hypercube xi ∈ [-500, 500], for all i = 1, …, d.
# Global minimum 0 at (420.9687, 420.9687, ..., 420.9687)
def schwefel(xx):
    d = len(xx)
    xx = np.array(xx)
    sum1 = np.sum(xx * np.sin(np.sqrt(np.abs(xx))))
    return 418.9829*d + sum1

# BOWL-SHAPED
# Perm function
# The function is usually evaluated on the hypercube xi ∈ [-d, d], for all i = 1, …, d.
# Global minimum 0 at (1, 1/2, 1/3, ..., 1/d)
def perm(xx, b=10):
    d = len(xx)
    xx = np.array(xx)

    outer = 0
    for i in range(d):
        inner = 0
        for j in range(d):
            inner = inner + ((j+1) + b) * (xx[j]**(i+1) - 1/(j+1)**(i+1))
        outer = outer + inner**2

    return outer

# BOWL-SHAPED
# Sphere on the domain [-5.12, 5.12]^dim, dim in {10, 20, 40}
# The Sphere function has d local minima except for the global one.
# It is continuous, convex and unimodal.
# The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
# Global minimum 0 at (0,0,...,0)
def sphere(xx):
    xx = np.array(xx)
    return np.sum(xx**2)

# PLATE-SHAPED
# Zakharov function
# The Zakharov function has no local minima except the global one.
# The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d.
# Global minimum 0 at (0,0,...,0)
def zakharov(xx):
    d = len(xx)
    xx = np.array(xx)
    sum1 = np.sum(xx**2)
    sum2 = 0
    for i in range(d):
        sum2 = sum2 + 0.5 * (i+1) * xx[i]
    return sum1 + sum2**2 + sum2**4

# VALLEY-SHAPED
# Rosenbrock on the domain [-2.048, 2.048]^dim, dim in {10, 20, 40}
# The Rosenbrock function, also referred to as the Valley or Banana function, is
# a popular test problem for gradient-based optimization algorithms.
# The function is unimodal, and the global minimum lies in a narrow, parabolic valley.
# The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d,
# although it may be restricted to the hypercube xi ∈ [-2.048, 2.048], for all i = 1, …, d.
# Global minimum 0 at (1,1,...,1)
def rosenbrock(xx):
    d = len(xx)
    xx = np.array(xx)
    sum1 = 0
    for i in range(d-1):
        sum1 = sum1 + 100 * (xx[i+1] - xx[i]**2)**2 + (xx[i] - 1)**2
    return sum1

# STEEP RIDGES
# Michalewicz function
# The Michalewicz function has d! local minima, and it is multimodal.
# The parameter m defines the steepness of the valleys and ridges;
# a larger m leads to a more difficult search. The recommended value of m is m = 10.
# The function is usually evaluated on the hypercube xi ∈ [0, π], for all i = 1, …, d.
# For d=2  global minimum -1.8013 at (2.20, 1.57)
# For d=5  global minimum -4.687658
# For d=10 global minimum -9.66015
def michalewicz(xx, m=10):
    d = len(xx)
    xx = np.array(xx)
    sum1 = 0.0
    for i in range(d):
        sum1 = sum1 + np.sin(xx[i-1]) * np.sin((i+1) * xx[i]**2 / pi)**(2 * m)
    return -sum1

# OTHER
# Perm function d, beta
# The function is usually evaluated on the hypercube xi ∈ [-d, d], for all i = 1, …, d.
# Global minimum 0 at (1,2,...,d)
def permdb(xx, b=0.5):
    d = len(xx)
    xx = np.array(xx)
    outer = 0
    for i in range(d):
        inner = 0
        for j in range(d):
            inner = inner + ((j+1)**(i+1) + b) * ((xx[j]/(j+1))**(i+1) - 1)
        outer = outer + inner**2
    return outer


# OTHER
# Styblinski-Tang on the domain [-5.00, 5.00]^dim, dim in {10, 20, 40}
# The function is usually evaluated on the hypercube xi ∈ [-5, 5], for all i = 1, …, d.
# Global minimum -39.6599d at (-2.903534,...,-2.903534)
def stybtang(xx):
    xx=np.array(xx)
    return 0.5 * np.sum(xx**4 - 16*xx**2 + 5*xx)


# Empty test
def do_nothing(**kwargs):
    return 0


if __name__ == '__main__':
    from dsopt import dsopt

    best_value, best_point = dsopt(grlee12,
                                   {'x': np.linspace(0.5, 2.5, 2001)},
                                   opt='min',
                                   max_evaluations=100,
                                   initial_sample_size=12,
                                   iterative_sample_size=8)

    print('best value: ', best_value)
    print('best point: ', best_point)

    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(grlee12,
    #                                         {'x': np.linspace(0.5, 2.5, 1000)})
    #     print(f'grlee12, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(holder,
    #                                         {'x': np.linspace(-10, 10, 1000),
    #                                          'y': np.linspace(-10, 10, 1000)})
    #     print(f'holder, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(levy13,
    #                                         {'x': np.linspace(-10, 10, 1000),
    #                                          'y': np.linspace(-10, 10, 1000)})
    #     print(f'levy13, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(shaffer2,
    #                                         {'x': np.linspace(-100, 100, 20000),
    #                                          'y': np.linspace(-100, 100, 20000)})
    #     print(f'shaffer2, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(eggholder,
    #                                         {'x': np.linspace(-512, 512, 100000),
    #                                          'y': np.linspace(-512, 512, 100000)})
    #     print(f'eggholder, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(dropwave,
    #                                         {'x': np.linspace(-5.12, 5.12, 5000),
    #                                          'y': np.linspace(-5.12, 5.12, 5000)})
    #     print(f'dropwave, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(crossit,
    #                                         {'x': np.linspace(-10, 10, 20000),
    #                                          'y': np.linspace(-10, 10, 20000)})
    #     print(f'crossit, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(bukin6,
    #                                         {'x': np.linspace(-15, -5, 10000),
    #                                          'y': np.linspace(-3, 3, 6000)})
    #     print(f'bukin6, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(camel6,
    #                                         {'x': np.linspace(-3, 3, 60000),
    #                                          'y': np.linspace(-2, 2, 40000)})
    #     print(f'camel6, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(easom,
    #                                         {'x': np.linspace(-100, 100, 200000),
    #                                          'y': np.linspace(-100, 100, 200000)})
    #     print(f'easom, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(beale,
    #                                         {'x': np.linspace(-4.5, 4.5, 9000),
    #                                          'y': np.linspace(-4.5, 4.5, 9000)})
    #     print(f'beale, run {i}, best value={best_value}, best_point={best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(branin,
    #                                         {'x': np.linspace(-5, 10, 150000),
    #                                          'y': np.linspace(0, 15, 150000)})
    #     print(f'branin, run {i}, best value={best_value}, best_point={best_point}')

    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(levy,
    #                                         {'xx': (20, np.linspace(-10, 10, 20000))})
    #     print(f'levy, run {i}, best value={best_value} for point: {best_point}')

    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(schwefel,
    #                                         {'xx': (20, np.linspace(-500, 500, 100000))})
    #     print(f'schwefel, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(perm,
    #                                         {'xx': (10, np.linspace(-10, 10, 20000))})
    #     print(f'perm, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(zakharov,
    #                                         {'xx': (20, np.linspace(-5, 10, 15000))})
    #     print(f'zakharov, run {i}, best value={best_value} for point: {best_point}')

    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(michalewicz,
    #                                         {'xx': (10, np.linspace(0, 3.1415, 32000))})
    #     print(f'michalewicz, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(10):
    #     best_value, best_point = dsopt_mipt(permdb,
    #                                         {'xx': (20, np.linspace(-20, 20, 40000))})
    #     print(f'permdb, run {i}, best value={best_value} for point: {best_point}')

    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(ackley,
    #                                         {'xx': (200, np.linspace(-5, 10, 15000))})
    #     print(f'ackley200, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(rosenbrock,
    #                                         {'xx': (10, np.linspace(-2.048, 2.048, 5000))})
    #     print(f'rosenbrock10, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(rosenbrock,
    #                                         {'xx': (20, np.linspace(-2.04, 2.048, 5000))})
    #     print(f'rosenbrock20, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(rosenbrock,
    #                                         {'xx': (40, np.linspace(-2.048, 2.048, 5000))})
    #     print(f'rosenbrock40, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(rastrigin,
    #                                         {'xx': (10, np.linspace(-5.12, 5.12, 10000))})
    #     print(f'rastrigin10, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(rastrigin,
    #                                         {'xx': (20, np.linspace(-5.12, 5.12, 10000))})
    #     print(f'rastrigin20, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(rastrigin,
    #                                         {'xx': (40, np.linspace(-5.12, 5.12, 5000))})
    #     print(f'rastrigin40, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(sphere,
    #                                         {'xx': (10, np.linspace(-5.12, 5.12, 10000))})
    #     print(f'sphere10, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(sphere,
    #                                         {'xx': (20, np.linspace(-5.12, 5.12, 10000))})
    #     print(f'sphere20, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(sphere,
    #                                         {'xx': (40, np.linspace(-5.12, 5.12, 10000))})
    #     print(f'sphere40, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(stybtang,
    #                                         {'xx': (10, np.linspace(-5, 5, 10000))})
    #     print(f'stybtang10, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(stybtang,
    #                                         {'xx': (20, np.linspace(-5, 5, 10000))})
    #     print(f'stybtang20, run {i}, best value={best_value} for point: {best_point}')
    #
    # for i in range(5):
    #     best_value, best_point = dsopt_mipt(stybtang,
    #                                         {'xx': (40, np.linspace(-5, 5, 10000))})
    #     print(f'stybtang40, run {i}, best value={best_value} for point: {best_point}')

    # best_value, best_parameters = dsopt(do_nothing,
    #                                     {'a': [0,1,2,3],
    #                                      'b': (['up', 'down'], 'c'),
    #                                      'c': (10, [7,8,9]),
    #                                      'd': (5, ['left', 'right'], 'c')})

