import sys
sys.path=['..']+sys.path

import numpy as np
import math
from geneticalgorithm import geneticalgorithm as ga


def f(X):
    dim = len(X)
    OF = np.sum((X**2)-10*np.cos(2*math.pi*X)+10)
    return OF


def test_rastrigin_initialized():
    parameters={'max_num_iteration': 100,
                'population_size':2000,
                'mutation_probability':0.1,
                'elit_ratio': 0.02,
                'crossover_probability': 0.5,
                'parents_portion': 0.3,
                'crossover_type':'two_point',
                'max_iteration_without_improv':None,
                'multiprocessing_ncpus': 4,
                'multiprocessing_engine': None,
                }

    dimension=200
    varbound = np.array([[-5.12, 5.12]]*dimension)

    model = ga(function=f, dimension=dimension, variable_type='real',
            variable_boundaries=varbound, algorithm_parameters=parameters)
    model.run(plot=True,initial_idv=np.random.random(size=dimension)*5.12-10.24)
    assert model.best_function < 1000


if __name__ == '__main__':
    test_rastrigin_initialized()
