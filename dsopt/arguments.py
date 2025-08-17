"""
Auxiliary methods for handling arguments of the expensive function.
"""
import numpy as np
import pandas as pd
import math
import copy


def analyze_arguments(arg_dict):
    """
    Goes through the dictionary of arguments for the expensive function,
    and returns the variables representing their various aspects:
    - dim:          total dimension of the search space
    - arg_type:     list of types for each dimension - 'q' for scalars and 'c' for categorical variables
    - arg_array:    list of names of arrays containing feasible values for each dimension
    - arg_len:      numpy array of lengths of arrays containing feasible values for each dimension

    :param arg_dict:    dictionary of expensive function arguments
    :return:            None
    """

    # first pass to pick up the dimension
    dim = 0
    for key, value in arg_dict.items():
        if type(value) is tuple:
            if type(value[0]) is list or type(value[0]) is np.ndarray:
                # value is ([array], 'c'), a categorical scalar
                dim += 1
            else:
                # value is (K, [array]) or (K, [array], 'c'), a numerical or a categorical array
                dim += value[0]
        else:
            # value is [array_of_feasible_values], a numerical scalar
            dim += 1

    arg_array = np.empty(dim, dtype=object)
    arg_len = np.empty(dim, dtype=int)
    arg_type = np.empty(dim, dtype='<U1')

    # second pass to pick up other arguments
    index = 0
    for key, value in arg_dict.items():
        if type(value) is tuple:
            if type(value[0]) is list or type(value[0]) is np.ndarray:
                # value is ([array], 'c'), a categorical scalar
                arg_array[index] = np.array(value[0])
                arg_len[index] = len(value[0])
                arg_type[index] = 'c'
                index += 1
            else:
                # value is (K, [array]) or (K, [array], 'c'), a numerical or a categorical array
                K = value[0]
                for i in range(K):
                    arg_array[index + i] = np.array(value[1])
                    arg_len[index + i] = len(value[1])
                if len(value) == 2:                     # numerical array
                    for i in range(K):
                        arg_type[index + i] = 'q'
                else:                                   # categorical array
                    for i in range(K):
                        arg_type[index + i] = 'c'
                index += K
        else:
            # value is [array_of_feasible_values], a numerical scalar
            arg_array[index] = np.array(value)
            arg_len[index] = len(value)
            arg_type[index] = 'q'
            index += 1

    return dim, arg_type, arg_array, arg_len


def hypercube_to_arguments(hyperpoints, arg_dict):
    """
    Auxiliary method to convert an array representing points in the hypercube [0,1]^dim
    into a list of dictionaries with corresponding values of the arguments
    from the global dictionary of expensive function arguments.

    :param hyperpoints: array of points from the hypercube [0,1]^dim
    :param arg_dict:    dictionary of expensive function arguments
    :return:            list of dictionaries of values of expensive function arguments in the original shape
    """
    arg_values = [None] * len(hyperpoints)
    # arg_values = [arg_dict.copy()] * len(hyperpoints)
    # arg_values = np.empty(len(hyperpoints), dtype=object)

    for i in range(len(hyperpoints)):
        p = hyperpoints[i]
        arg_values[i] = copy.deepcopy(arg_dict)

        index = 0
        for key, value in arg_dict.items():
            if type(value) is tuple:
                if type(value[0]) is list or type(value[0]) is np.ndarray:
                    # value is ([array], 'c'), a categorical scalar
                    length = len(value[0])
                    arg_values[i][key] = value[0][math.floor(p[index] * length)]
                    index += 1
                else:
                    # value is (K, [array]) or (K, [array], 'c'), a numerical or a categorical array
                    length = len(value[1])
                    arg_values[i][key] = [value[1][math.floor(p[index+j] * length)] for j in range(value[0])]
                    index += value[0]
            else:
                # value is [array_of_feasible_values], a numerical scalar
                length = len(value)
                arg_values[i][key] = value[math.floor(p[index] * length)]
                index += 1

    return arg_values


#############################################################################################
# Auxiliary method to convert an array of points obtained by calling hypercube_to_arguments #
# into a flattened form compatible with the result of hypercube_to_sample                   #
#############################################################################################
def points_to_sample(points, dim, arg_type):
    sample = np.empty((len(points), dim), dtype=object)

    for i in range(len(points)):
        p = points[i]

        index = 0
        for key, value in p.items():
            if type(value) is list:
                for j in range(len(value)):
                    sample[i, index+j] = value[j]
                index += len(value)
            else:
                sample[i, index] = value
                index += 1

    df = pd.DataFrame(data=sample)
    for i in range(dim):
        if arg_type[i] == 'c':
            df[i] = df[i].astype('category')
        else:
            df[i] = df[i].astype('float')

    return df


################################################################################
# Auxiliary method to convert an array of points from the hypercube [0,1]^dim  #
# into corresponding values of the arguments from the list of argument arrays. #
################################################################################
def hypercube_to_sample(hyperpoints, dim, arg_type, arg_array, arg_len):
    """
    Returns a dataframe containing sample with expanded argument values for XGBoost training and prediction
    """
    sample = np.empty(hyperpoints.shape, dtype=object)

    # vectorization across each dimension separately
    for i in range(dim):
        indices = np.floor(hyperpoints[:, i] * arg_len[i]).astype(int)
        sample[:, i] = arg_array[i][indices]

    df = pd.DataFrame(data=sample)
    for i in range(dim):
        if arg_type[i] == 'c':
            df[i] = df[i].astype('category')
        else:
            df[i] = df[i].astype('float')

    return df

