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


def smac_coco_1_5(config: Configuration, seed: int = 0):
    return coco_categorical(1, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_1_10(config: Configuration, seed: int = 0):
    return coco_categorical(1, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_1_20(config: Configuration, seed: int = 0):
    return coco_categorical(1, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_1_40(config: Configuration, seed: int = 0):
    return coco_categorical(1, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_1_80(config: Configuration, seed: int = 0):
    return coco_categorical(1, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_1_160(config: Configuration, seed: int = 0):
    return coco_categorical(1, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_2_5(config: Configuration, seed: int = 0):
    return coco_categorical(2, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_2_10(config: Configuration, seed: int = 0):
    return coco_categorical(2, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_2_20(config: Configuration, seed: int = 0):
    return coco_categorical(2, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_2_40(config: Configuration, seed: int = 0):
    return coco_categorical(2, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_2_80(config: Configuration, seed: int = 0):
    return coco_categorical(2, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_2_160(config: Configuration, seed: int = 0):
    return coco_categorical(2, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_3_5(config: Configuration, seed: int = 0):
    return coco_categorical(3, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_3_10(config: Configuration, seed: int = 0):
    return coco_categorical(3, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_3_20(config: Configuration, seed: int = 0):
    return coco_categorical(3, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_3_40(config: Configuration, seed: int = 0):
    return coco_categorical(3, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_3_80(config: Configuration, seed: int = 0):
    return coco_categorical(3, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_3_160(config: Configuration, seed: int = 0):
    return coco_categorical(3, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_4_5(config: Configuration, seed: int = 0):
    return coco_categorical(4, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_4_10(config: Configuration, seed: int = 0):
    return coco_categorical(4, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_4_20(config: Configuration, seed: int = 0):
    return coco_categorical(4, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_4_40(config: Configuration, seed: int = 0):
    return coco_categorical(4, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_4_80(config: Configuration, seed: int = 0):
    return coco_categorical(4, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_4_160(config: Configuration, seed: int = 0):
    return coco_categorical(4, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_5_5(config: Configuration, seed: int = 0):
    return coco_categorical(5, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_5_10(config: Configuration, seed: int = 0):
    return coco_categorical(5, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_5_20(config: Configuration, seed: int = 0):
    return coco_categorical(5, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_5_40(config: Configuration, seed: int = 0):
    return coco_categorical(5, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_5_80(config: Configuration, seed: int = 0):
    return coco_categorical(5, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_5_160(config: Configuration, seed: int = 0):
    return coco_categorical(5, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_6_5(config: Configuration, seed: int = 0):
    return coco_categorical(6, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_6_10(config: Configuration, seed: int = 0):
    return coco_categorical(6, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_6_20(config: Configuration, seed: int = 0):
    return coco_categorical(6, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_6_40(config: Configuration, seed: int = 0):
    return coco_categorical(6, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_6_80(config: Configuration, seed: int = 0):
    return coco_categorical(6, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_6_160(config: Configuration, seed: int = 0):
    return coco_categorical(6, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_7_5(config: Configuration, seed: int = 0):
    return coco_categorical(7, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_7_10(config: Configuration, seed: int = 0):
    return coco_categorical(7, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_7_20(config: Configuration, seed: int = 0):
    return coco_categorical(7, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_7_40(config: Configuration, seed: int = 0):
    return coco_categorical(7, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_7_80(config: Configuration, seed: int = 0):
    return coco_categorical(7, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_7_160(config: Configuration, seed: int = 0):
    return coco_categorical(7, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_8_5(config: Configuration, seed: int = 0):
    return coco_categorical(8, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_8_10(config: Configuration, seed: int = 0):
    return coco_categorical(8, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_8_20(config: Configuration, seed: int = 0):
    return coco_categorical(8, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_8_40(config: Configuration, seed: int = 0):
    return coco_categorical(8, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_8_80(config: Configuration, seed: int = 0):
    return coco_categorical(8, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_8_160(config: Configuration, seed: int = 0):
    return coco_categorical(8, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_9_5(config: Configuration, seed: int = 0):
    return coco_categorical(9, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_9_10(config: Configuration, seed: int = 0):
    return coco_categorical(9, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_9_20(config: Configuration, seed: int = 0):
    return coco_categorical(9, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_9_40(config: Configuration, seed: int = 0):
    return coco_categorical(9, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_9_80(config: Configuration, seed: int = 0):
    return coco_categorical(9, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_9_160(config: Configuration, seed: int = 0):
    return coco_categorical(9, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_10_5(config: Configuration, seed: int = 0):
    return coco_categorical(10, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_10_10(config: Configuration, seed: int = 0):
    return coco_categorical(10, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_10_20(config: Configuration, seed: int = 0):
    return coco_categorical(10, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_10_40(config: Configuration, seed: int = 0):
    return coco_categorical(10, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_10_80(config: Configuration, seed: int = 0):
    return coco_categorical(10, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_10_160(config: Configuration, seed: int = 0):
    return coco_categorical(10, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_11_5(config: Configuration, seed: int = 0):
    return coco_categorical(11, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_11_10(config: Configuration, seed: int = 0):
    return coco_categorical(11, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_11_20(config: Configuration, seed: int = 0):
    return coco_categorical(11, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_11_40(config: Configuration, seed: int = 0):
    return coco_categorical(11, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_11_80(config: Configuration, seed: int = 0):
    return coco_categorical(11, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_11_160(config: Configuration, seed: int = 0):
    return coco_categorical(11, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_12_5(config: Configuration, seed: int = 0):
    return coco_categorical(12, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_12_10(config: Configuration, seed: int = 0):
    return coco_categorical(12, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_12_20(config: Configuration, seed: int = 0):
    return coco_categorical(12, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_12_40(config: Configuration, seed: int = 0):
    return coco_categorical(12, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_12_80(config: Configuration, seed: int = 0):
    return coco_categorical(12, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_12_160(config: Configuration, seed: int = 0):
    return coco_categorical(12, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_13_5(config: Configuration, seed: int = 0):
    return coco_categorical(13, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_13_10(config: Configuration, seed: int = 0):
    return coco_categorical(13, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_13_20(config: Configuration, seed: int = 0):
    return coco_categorical(13, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_13_40(config: Configuration, seed: int = 0):
    return coco_categorical(13, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_13_80(config: Configuration, seed: int = 0):
    return coco_categorical(13, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_13_160(config: Configuration, seed: int = 0):
    return coco_categorical(13, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_14_5(config: Configuration, seed: int = 0):
    return coco_categorical(14, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_14_10(config: Configuration, seed: int = 0):
    return coco_categorical(14, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_14_20(config: Configuration, seed: int = 0):
    return coco_categorical(14, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_14_40(config: Configuration, seed: int = 0):
    return coco_categorical(14, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_14_80(config: Configuration, seed: int = 0):
    return coco_categorical(14, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_14_160(config: Configuration, seed: int = 0):
    return coco_categorical(14, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_15_5(config: Configuration, seed: int = 0):
    return coco_categorical(15, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_15_10(config: Configuration, seed: int = 0):
    return coco_categorical(15, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_15_20(config: Configuration, seed: int = 0):
    return coco_categorical(15, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_15_40(config: Configuration, seed: int = 0):
    return coco_categorical(15, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_15_80(config: Configuration, seed: int = 0):
    return coco_categorical(15, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_15_160(config: Configuration, seed: int = 0):
    return coco_categorical(15, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_16_5(config: Configuration, seed: int = 0):
    return coco_categorical(16, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_16_10(config: Configuration, seed: int = 0):
    return coco_categorical(16, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_16_20(config: Configuration, seed: int = 0):
    return coco_categorical(16, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_16_40(config: Configuration, seed: int = 0):
    return coco_categorical(16, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_16_80(config: Configuration, seed: int = 0):
    return coco_categorical(16, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_16_160(config: Configuration, seed: int = 0):
    return coco_categorical(16, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_17_5(config: Configuration, seed: int = 0):
    return coco_categorical(17, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_17_10(config: Configuration, seed: int = 0):
    return coco_categorical(17, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_17_20(config: Configuration, seed: int = 0):
    return coco_categorical(17, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_17_40(config: Configuration, seed: int = 0):
    return coco_categorical(17, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_17_80(config: Configuration, seed: int = 0):
    return coco_categorical(17, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_17_160(config: Configuration, seed: int = 0):
    return coco_categorical(17, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_18_5(config: Configuration, seed: int = 0):
    return coco_categorical(18, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_18_10(config: Configuration, seed: int = 0):
    return coco_categorical(18, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_18_20(config: Configuration, seed: int = 0):
    return coco_categorical(18, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_18_40(config: Configuration, seed: int = 0):
    return coco_categorical(18, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_18_80(config: Configuration, seed: int = 0):
    return coco_categorical(18, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_18_160(config: Configuration, seed: int = 0):
    return coco_categorical(18, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_19_5(config: Configuration, seed: int = 0):
    return coco_categorical(19, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_19_10(config: Configuration, seed: int = 0):
    return coco_categorical(19, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_19_20(config: Configuration, seed: int = 0):
    return coco_categorical(19, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_19_40(config: Configuration, seed: int = 0):
    return coco_categorical(19, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_19_80(config: Configuration, seed: int = 0):
    return coco_categorical(19, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_19_160(config: Configuration, seed: int = 0):
    return coco_categorical(19, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_20_5(config: Configuration, seed: int = 0):
    return coco_categorical(20, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_20_10(config: Configuration, seed: int = 0):
    return coco_categorical(20, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_20_20(config: Configuration, seed: int = 0):
    return coco_categorical(20, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_20_40(config: Configuration, seed: int = 0):
    return coco_categorical(20, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_20_80(config: Configuration, seed: int = 0):
    return coco_categorical(20, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_20_160(config: Configuration, seed: int = 0):
    return coco_categorical(20, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_21_5(config: Configuration, seed: int = 0):
    return coco_categorical(21, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_21_10(config: Configuration, seed: int = 0):
    return coco_categorical(21, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_21_20(config: Configuration, seed: int = 0):
    return coco_categorical(21, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_21_40(config: Configuration, seed: int = 0):
    return coco_categorical(21, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_21_80(config: Configuration, seed: int = 0):
    return coco_categorical(21, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_21_160(config: Configuration, seed: int = 0):
    return coco_categorical(21, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_22_5(config: Configuration, seed: int = 0):
    return coco_categorical(22, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_22_10(config: Configuration, seed: int = 0):
    return coco_categorical(22, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_22_20(config: Configuration, seed: int = 0):
    return coco_categorical(22, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_22_40(config: Configuration, seed: int = 0):
    return coco_categorical(22, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_22_80(config: Configuration, seed: int = 0):
    return coco_categorical(22, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_22_160(config: Configuration, seed: int = 0):
    return coco_categorical(22, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_23_5(config: Configuration, seed: int = 0):
    return coco_categorical(23, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_23_10(config: Configuration, seed: int = 0):
    return coco_categorical(23, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_23_20(config: Configuration, seed: int = 0):
    return coco_categorical(23, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_23_40(config: Configuration, seed: int = 0):
    return coco_categorical(23, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_23_80(config: Configuration, seed: int = 0):
    return coco_categorical(23, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_23_160(config: Configuration, seed: int = 0):
    return coco_categorical(23, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_coco_24_5(config: Configuration, seed: int = 0):
    return coco_categorical(24, 5, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_24_10(config: Configuration, seed: int = 0):
    return coco_categorical(24, 10, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_24_20(config: Configuration, seed: int = 0):
    return coco_categorical(24, 20, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_24_40(config: Configuration, seed: int = 0):
    return coco_categorical(24, 40, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_24_80(config: Configuration, seed: int = 0):
    return coco_categorical(24, 80, list(config.values())[:-2], config['ca'], config['cb'])

def smac_coco_24_160(config: Configuration, seed: int = 0):
    return coco_categorical(24, 160, list(config.values())[:-2], config['ca'], config['cb'])


def smac_kw_villa(config: Configuration, seed: int = 0):
    return simulate_desert_villa(glazing_open_facade=config["glazing_open_facade"],
                                 shading_open_facade=config["shading_open_facade"],
                                 glazing_closed_facade=config["glazing_closed_facade"],
                                 wwr_front=config["wwr_front"],
                                 exterior_wall=config["exterior_wall"],
                                 insulation_thickness=config["insulation_thickness"])

