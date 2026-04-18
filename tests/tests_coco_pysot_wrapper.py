import cocoex
from ConfigSpace import Configuration
from kw_villa import simulate_desert_villa

def coco_categorical(f_index,
                     dimension,
                     args,
                     cat_first,
                     cat_second):
    """
    Selects the appropriate COCO testbed function from the bbob-mixint suite,
    and computes its value for given integer- and real-valued arguments.

    :param f_index:     function index, from 1 to 24
    :param dimension:   problem dimension, one of 5, 10, 20, 40, 80, 160
    :param args:        array of arguments for the corresponding instance of COCO function
    :param cat_first:   categorical argument from {0, 1, 2}
    :param cat_second:  categorical argument from {0, ..., 4}
    :return:
    """

    # instance index
    instance = 5*cat_first + cat_second + 1

    # get the appropriate function
    suite = cocoex.Suite("bbob-mixint", "", "")
    problem = suite.get_problem_by_function_dimension_instance(f_index, dimension, instance)

    # compute the function value
    value = problem(args)

    # free the problem
    problem.free()

    # return the function value
    return value


def pysot_coco_1_5(X):
    return coco_categorical(1, 5, X[:-2], X[-2], X[-1])

def pysot_coco_1_10(X):
    return coco_categorical(1, 10, X[:-2], X[-2], X[-1])

def pysot_coco_1_20(X):
    return coco_categorical(1, 20, X[:-2], X[-2], X[-1])

def pysot_coco_1_40(X):
    return coco_categorical(1, 40, X[:-2], X[-2], X[-1])

def pysot_coco_1_80(X):
    return coco_categorical(1, 80, X[:-2], X[-2], X[-1])

def pysot_coco_1_160(X):
    return coco_categorical(1, 160, X[:-2], X[-2], X[-1])


def pysot_coco_2_5(X):
    return coco_categorical(2, 5, X[:-2], X[-2], X[-1])

def pysot_coco_2_10(X):
    return coco_categorical(2, 10, X[:-2], X[-2], X[-1])

def pysot_coco_2_20(X):
    return coco_categorical(2, 20, X[:-2], X[-2], X[-1])

def pysot_coco_2_40(X):
    return coco_categorical(2, 40, X[:-2], X[-2], X[-1])

def pysot_coco_2_80(X):
    return coco_categorical(2, 80, X[:-2], X[-2], X[-1])

def pysot_coco_2_160(X):
    return coco_categorical(2, 160, X[:-2], X[-2], X[-1])


def pysot_coco_3_5(X):
    return coco_categorical(3, 5, X[:-2], X[-2], X[-1])

def pysot_coco_3_10(X):
    return coco_categorical(3, 10, X[:-2], X[-2], X[-1])

def pysot_coco_3_20(X):
    return coco_categorical(3, 20, X[:-2], X[-2], X[-1])

def pysot_coco_3_40(X):
    return coco_categorical(3, 40, X[:-2], X[-2], X[-1])

def pysot_coco_3_80(X):
    return coco_categorical(3, 80, X[:-2], X[-2], X[-1])

def pysot_coco_3_160(X):
    return coco_categorical(3, 160, X[:-2], X[-2], X[-1])


def pysot_coco_4_5(X):
    return coco_categorical(4, 5, X[:-2], X[-2], X[-1])

def pysot_coco_4_10(X):
    return coco_categorical(4, 10, X[:-2], X[-2], X[-1])

def pysot_coco_4_20(X):
    return coco_categorical(4, 20, X[:-2], X[-2], X[-1])

def pysot_coco_4_40(X):
    return coco_categorical(4, 40, X[:-2], X[-2], X[-1])

def pysot_coco_4_80(X):
    return coco_categorical(4, 80, X[:-2], X[-2], X[-1])

def pysot_coco_4_160(X):
    return coco_categorical(4, 160, X[:-2], X[-2], X[-1])


def pysot_coco_5_5(X):
    return coco_categorical(5, 5, X[:-2], X[-2], X[-1])

def pysot_coco_5_10(X):
    return coco_categorical(5, 10, X[:-2], X[-2], X[-1])

def pysot_coco_5_20(X):
    return coco_categorical(5, 20, X[:-2], X[-2], X[-1])

def pysot_coco_5_40(X):
    return coco_categorical(5, 40, X[:-2], X[-2], X[-1])

def pysot_coco_5_80(X):
    return coco_categorical(5, 80, X[:-2], X[-2], X[-1])

def pysot_coco_5_160(X):
    return coco_categorical(5, 160, X[:-2], X[-2], X[-1])


def pysot_coco_6_5(X):
    return coco_categorical(6, 5, X[:-2], X[-2], X[-1])

def pysot_coco_6_10(X):
    return coco_categorical(6, 10, X[:-2], X[-2], X[-1])

def pysot_coco_6_20(X):
    return coco_categorical(6, 20, X[:-2], X[-2], X[-1])

def pysot_coco_6_40(X):
    return coco_categorical(6, 40, X[:-2], X[-2], X[-1])

def pysot_coco_6_80(X):
    return coco_categorical(6, 80, X[:-2], X[-2], X[-1])

def pysot_coco_6_160(X):
    return coco_categorical(6, 160, X[:-2], X[-2], X[-1])


def pysot_coco_7_5(X):
    return coco_categorical(7, 5, X[:-2], X[-2], X[-1])

def pysot_coco_7_10(X):
    return coco_categorical(7, 10, X[:-2], X[-2], X[-1])

def pysot_coco_7_20(X):
    return coco_categorical(7, 20, X[:-2], X[-2], X[-1])

def pysot_coco_7_40(X):
    return coco_categorical(7, 40, X[:-2], X[-2], X[-1])

def pysot_coco_7_80(X):
    return coco_categorical(7, 80, X[:-2], X[-2], X[-1])

def pysot_coco_7_160(X):
    return coco_categorical(7, 160, X[:-2], X[-2], X[-1])


def pysot_coco_8_5(X):
    return coco_categorical(8, 5, X[:-2], X[-2], X[-1])

def pysot_coco_8_10(X):
    return coco_categorical(8, 10, X[:-2], X[-2], X[-1])

def pysot_coco_8_20(X):
    return coco_categorical(8, 20, X[:-2], X[-2], X[-1])

def pysot_coco_8_40(X):
    return coco_categorical(8, 40, X[:-2], X[-2], X[-1])

def pysot_coco_8_80(X):
    return coco_categorical(8, 80, X[:-2], X[-2], X[-1])

def pysot_coco_8_160(X):
    return coco_categorical(8, 160, X[:-2], X[-2], X[-1])


def pysot_coco_9_5(X):
    return coco_categorical(9, 5, X[:-2], X[-2], X[-1])

def pysot_coco_9_10(X):
    return coco_categorical(9, 10, X[:-2], X[-2], X[-1])

def pysot_coco_9_20(X):
    return coco_categorical(9, 20, X[:-2], X[-2], X[-1])

def pysot_coco_9_40(X):
    return coco_categorical(9, 40, X[:-2], X[-2], X[-1])

def pysot_coco_9_80(X):
    return coco_categorical(9, 80, X[:-2], X[-2], X[-1])

def pysot_coco_9_160(X):
    return coco_categorical(9, 160, X[:-2], X[-2], X[-1])


def pysot_coco_10_5(X):
    return coco_categorical(10, 5, X[:-2], X[-2], X[-1])

def pysot_coco_10_10(X):
    return coco_categorical(10, 10, X[:-2], X[-2], X[-1])

def pysot_coco_10_20(X):
    return coco_categorical(10, 20, X[:-2], X[-2], X[-1])

def pysot_coco_10_40(X):
    return coco_categorical(10, 40, X[:-2], X[-2], X[-1])

def pysot_coco_10_80(X):
    return coco_categorical(10, 80, X[:-2], X[-2], X[-1])

def pysot_coco_10_160(X):
    return coco_categorical(10, 160, X[:-2], X[-2], X[-1])


def pysot_coco_11_5(X):
    return coco_categorical(11, 5, X[:-2], X[-2], X[-1])

def pysot_coco_11_10(X):
    return coco_categorical(11, 10, X[:-2], X[-2], X[-1])

def pysot_coco_11_20(X):
    return coco_categorical(11, 20, X[:-2], X[-2], X[-1])

def pysot_coco_11_40(X):
    return coco_categorical(11, 40, X[:-2], X[-2], X[-1])

def pysot_coco_11_80(X):
    return coco_categorical(11, 80, X[:-2], X[-2], X[-1])

def pysot_coco_11_160(X):
    return coco_categorical(11, 160, X[:-2], X[-2], X[-1])


def pysot_coco_12_5(X):
    return coco_categorical(12, 5, X[:-2], X[-2], X[-1])

def pysot_coco_12_10(X):
    return coco_categorical(12, 10, X[:-2], X[-2], X[-1])

def pysot_coco_12_20(X):
    return coco_categorical(12, 20, X[:-2], X[-2], X[-1])

def pysot_coco_12_40(X):
    return coco_categorical(12, 40, X[:-2], X[-2], X[-1])

def pysot_coco_12_80(X):
    return coco_categorical(12, 80, X[:-2], X[-2], X[-1])

def pysot_coco_12_160(X):
    return coco_categorical(12, 160, X[:-2], X[-2], X[-1])


def pysot_coco_13_5(X):
    return coco_categorical(13, 5, X[:-2], X[-2], X[-1])

def pysot_coco_13_10(X):
    return coco_categorical(13, 10, X[:-2], X[-2], X[-1])

def pysot_coco_13_20(X):
    return coco_categorical(13, 20, X[:-2], X[-2], X[-1])

def pysot_coco_13_40(X):
    return coco_categorical(13, 40, X[:-2], X[-2], X[-1])

def pysot_coco_13_80(X):
    return coco_categorical(13, 80, X[:-2], X[-2], X[-1])

def pysot_coco_13_160(X):
    return coco_categorical(13, 160, X[:-2], X[-2], X[-1])


def pysot_coco_14_5(X):
    return coco_categorical(14, 5, X[:-2], X[-2], X[-1])

def pysot_coco_14_10(X):
    return coco_categorical(14, 10, X[:-2], X[-2], X[-1])

def pysot_coco_14_20(X):
    return coco_categorical(14, 20, X[:-2], X[-2], X[-1])

def pysot_coco_14_40(X):
    return coco_categorical(14, 40, X[:-2], X[-2], X[-1])

def pysot_coco_14_80(X):
    return coco_categorical(14, 80, X[:-2], X[-2], X[-1])

def pysot_coco_14_160(X):
    return coco_categorical(14, 160, X[:-2], X[-2], X[-1])


def pysot_coco_15_5(X):
    return coco_categorical(15, 5, X[:-2], X[-2], X[-1])

def pysot_coco_15_10(X):
    return coco_categorical(15, 10, X[:-2], X[-2], X[-1])

def pysot_coco_15_20(X):
    return coco_categorical(15, 20, X[:-2], X[-2], X[-1])

def pysot_coco_15_40(X):
    return coco_categorical(15, 40, X[:-2], X[-2], X[-1])

def pysot_coco_15_80(X):
    return coco_categorical(15, 80, X[:-2], X[-2], X[-1])

def pysot_coco_15_160(X):
    return coco_categorical(15, 160, X[:-2], X[-2], X[-1])


def pysot_coco_16_5(X):
    return coco_categorical(16, 5, X[:-2], X[-2], X[-1])

def pysot_coco_16_10(X):
    return coco_categorical(16, 10, X[:-2], X[-2], X[-1])

def pysot_coco_16_20(X):
    return coco_categorical(16, 20, X[:-2], X[-2], X[-1])

def pysot_coco_16_40(X):
    return coco_categorical(16, 40, X[:-2], X[-2], X[-1])

def pysot_coco_16_80(X):
    return coco_categorical(16, 80, X[:-2], X[-2], X[-1])

def pysot_coco_16_160(X):
    return coco_categorical(16, 160, X[:-2], X[-2], X[-1])


def pysot_coco_17_5(X):
    return coco_categorical(17, 5, X[:-2], X[-2], X[-1])

def pysot_coco_17_10(X):
    return coco_categorical(17, 10, X[:-2], X[-2], X[-1])

def pysot_coco_17_20(X):
    return coco_categorical(17, 20, X[:-2], X[-2], X[-1])

def pysot_coco_17_40(X):
    return coco_categorical(17, 40, X[:-2], X[-2], X[-1])

def pysot_coco_17_80(X):
    return coco_categorical(17, 80, X[:-2], X[-2], X[-1])

def pysot_coco_17_160(X):
    return coco_categorical(17, 160, X[:-2], X[-2], X[-1])


def pysot_coco_18_5(X):
    return coco_categorical(18, 5, X[:-2], X[-2], X[-1])

def pysot_coco_18_10(X):
    return coco_categorical(18, 10, X[:-2], X[-2], X[-1])

def pysot_coco_18_20(X):
    return coco_categorical(18, 20, X[:-2], X[-2], X[-1])

def pysot_coco_18_40(X):
    return coco_categorical(18, 40, X[:-2], X[-2], X[-1])

def pysot_coco_18_80(X):
    return coco_categorical(18, 80, X[:-2], X[-2], X[-1])

def pysot_coco_18_160(X):
    return coco_categorical(18, 160, X[:-2], X[-2], X[-1])


def pysot_coco_19_5(X):
    return coco_categorical(19, 5, X[:-2], X[-2], X[-1])

def pysot_coco_19_10(X):
    return coco_categorical(19, 10, X[:-2], X[-2], X[-1])

def pysot_coco_19_20(X):
    return coco_categorical(19, 20, X[:-2], X[-2], X[-1])

def pysot_coco_19_40(X):
    return coco_categorical(19, 40, X[:-2], X[-2], X[-1])

def pysot_coco_19_80(X):
    return coco_categorical(19, 80, X[:-2], X[-2], X[-1])

def pysot_coco_19_160(X):
    return coco_categorical(19, 160, X[:-2], X[-2], X[-1])


def pysot_coco_20_5(X):
    return coco_categorical(20, 5, X[:-2], X[-2], X[-1])

def pysot_coco_20_10(X):
    return coco_categorical(20, 10, X[:-2], X[-2], X[-1])

def pysot_coco_20_20(X):
    return coco_categorical(20, 20, X[:-2], X[-2], X[-1])

def pysot_coco_20_40(X):
    return coco_categorical(20, 40, X[:-2], X[-2], X[-1])

def pysot_coco_20_80(X):
    return coco_categorical(20, 80, X[:-2], X[-2], X[-1])

def pysot_coco_20_160(X):
    return coco_categorical(20, 160, X[:-2], X[-2], X[-1])


def pysot_coco_21_5(X):
    return coco_categorical(21, 5, X[:-2], X[-2], X[-1])

def pysot_coco_21_10(X):
    return coco_categorical(21, 10, X[:-2], X[-2], X[-1])

def pysot_coco_21_20(X):
    return coco_categorical(21, 20, X[:-2], X[-2], X[-1])

def pysot_coco_21_40(X):
    return coco_categorical(21, 40, X[:-2], X[-2], X[-1])

def pysot_coco_21_80(X):
    return coco_categorical(21, 80, X[:-2], X[-2], X[-1])

def pysot_coco_21_160(X):
    return coco_categorical(21, 160, X[:-2], X[-2], X[-1])


def pysot_coco_22_5(X):
    return coco_categorical(22, 5, X[:-2], X[-2], X[-1])

def pysot_coco_22_10(X):
    return coco_categorical(22, 10, X[:-2], X[-2], X[-1])

def pysot_coco_22_20(X):
    return coco_categorical(22, 20, X[:-2], X[-2], X[-1])

def pysot_coco_22_40(X):
    return coco_categorical(22, 40, X[:-2], X[-2], X[-1])

def pysot_coco_22_80(X):
    return coco_categorical(22, 80, X[:-2], X[-2], X[-1])

def pysot_coco_22_160(X):
    return coco_categorical(22, 160, X[:-2], X[-2], X[-1])


def pysot_coco_23_5(X):
    return coco_categorical(23, 5, X[:-2], X[-2], X[-1])

def pysot_coco_23_10(X):
    return coco_categorical(23, 10, X[:-2], X[-2], X[-1])

def pysot_coco_23_20(X):
    return coco_categorical(23, 20, X[:-2], X[-2], X[-1])

def pysot_coco_23_40(X):
    return coco_categorical(23, 40, X[:-2], X[-2], X[-1])

def pysot_coco_23_80(X):
    return coco_categorical(23, 80, X[:-2], X[-2], X[-1])

def pysot_coco_23_160(X):
    return coco_categorical(23, 160, X[:-2], X[-2], X[-1])


def pysot_coco_24_5(X):
    return coco_categorical(24, 5, X[:-2], X[-2], X[-1])

def pysot_coco_24_10(X):
    return coco_categorical(24, 10, X[:-2], X[-2], X[-1])

def pysot_coco_24_20(X):
    return coco_categorical(24, 20, X[:-2], X[-2], X[-1])

def pysot_coco_24_40(X):
    return coco_categorical(24, 40, X[:-2], X[-2], X[-1])

def pysot_coco_24_80(X):
    return coco_categorical(24, 80, X[:-2], X[-2], X[-1])

def pysot_coco_24_160(X):
    return coco_categorical(24, 160, X[:-2], X[-2], X[-1])


def pysot_kw_villa(config):
    return simulate_desert_villa(glazing_open_facade=config["glazing_open_facade"],
                                 shading_open_facade=config["shading_open_facade"],
                                 glazing_closed_facade=config["glazing_closed_facade"],
                                 wwr_front=config["wwr_front"],
                                 exterior_wall=config["exterior_wall"],
                                 insulation_thickness=config["insulation_thickness"])

