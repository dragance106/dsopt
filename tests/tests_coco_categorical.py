"""
This file builds upon the mixed integer suite of COCO testbed functions (bbob-mixint)
to create their categorical versions that have all of integer, continuous and categorical arguments.
Namely, bbob-mixint consists of 24 functions, which come in dimensions 5, 10, 20, 40, 80 and 160.
Besides dimension, each function has 15 instances, indexed from 1-15,
where the instance number only influences the starting seed for the random number generator.
Hence changing the instance number modifies the test function in a manner
relatively similar to changing glazing type or wall construction in a building energy model.
Each test function of dimension d from bbox-mixint suite has:
- d/5 integer variables from {0, 1}
- d/5 integer variables from {0, 1, 2, 3}
- d/5 integer variables from {0, ..., 7}
- d/5 integer variables from {0, ..., 15}
- d/5 real variables from [-5, +5]

Categorization is performed with two additional arguments a and b
which serve to select one of 15 instances for each function.
The argument a is selected from the set {0, 1, 2},
the argument b is selected from the set {0, ..., 4},
and then one selects the instance 5a+b+1 for actual evaluation.

Functions 15--24 are multi-modal, so one should pay most attention to them in dsopt paper.

See https://numbbo.github.io/coco/testsuites/bbob-mixint, https://coco-platform.org/ and
https://numbbo.github.io/gforge/preliminary-bbob-mixint-documentation/bbob-mixint-doc.pdf
for function definitions.
"""
import cocoex

def coco_categorical(f_index,
                     dimension,
                     int_args2,
                     int_args4,
                     int_args8,
                     int_args16,
                     real_args,
                     cat_first,
                     cat_second):
    """
    Selects the appropriate COCO testbed function from the bbob-mixint suite,
    and computes its value for given integer- and real-valued arguments.

    :param f_index:     function index, from 1 to 24
    :param dimension:   problem dimension, one of 5, 10, 20, 40, 80, 160
    :param int_args2:   array of dimension/5 integer arguments from {0, 1}
    :param int_args4:   array of dimension/5 integer arguments from {0, 1, 2, 3}
    :param int_args8:   array of dimension/5 integer arguments from {0, ..., 7}
    :param int_args16:  array of dimension/5 integer arguments from {0, ..., 15}
    :param real_args:   array of dimension/5 real arguments from [-5, +5]
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
    value = problem(int_args2 + int_args4 + int_args8 + int_args16 + real_args)

    # free the problem
    problem.free()

    # return the function value
    return value

def coco_1_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(1, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_1_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(1, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_1_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(1, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_1_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(1, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_1_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(1, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_1_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(1, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_2_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(2, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_2_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(2, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_2_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(2, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_2_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(2, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_2_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(2, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_2_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(2, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_3_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(3, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_3_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(3, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_3_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(3, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_3_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(3, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_3_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(3, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_3_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(3, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_4_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(4, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_4_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(4, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_4_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(4, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_4_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(4, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_4_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(4, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_4_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(4, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_5_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(5, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_5_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(5, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_5_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(5, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_5_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(5, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_5_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(5, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_5_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(5, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_6_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(6, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_6_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(6, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_6_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(6, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_6_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(6, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_6_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(6, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_6_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(6, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_7_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(7, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_7_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(7, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_7_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(7, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_7_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(7, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_7_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(7, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_7_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(7, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_8_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(8, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_8_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(8, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_8_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(8, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_8_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(8, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_8_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(8, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_8_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(8, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_9_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(9, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_9_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(9, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_9_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(9, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_9_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(9, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_9_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(9, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_9_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(9, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_10_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(10, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_10_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(10, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_10_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(10, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_10_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(10, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_10_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(10, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_10_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(10, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_11_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(11, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_11_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(11, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_11_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(11, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_11_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(11, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_11_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(11, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_11_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(11, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_12_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(12, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_12_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(12, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_12_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(12, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_12_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(12, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_12_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(12, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_12_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(12, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_13_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(13, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_13_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(13, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_13_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(13, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_13_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(13, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_13_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(13, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_13_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(13, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_14_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(14, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_14_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(14, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_14_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(14, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_14_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(14, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_14_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(14, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_14_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(14, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_15_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(15, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_15_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(15, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_15_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(15, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_15_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(15, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_15_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(15, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_15_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(15, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_16_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(16, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_16_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(16, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_16_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(16, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_16_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(16, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_16_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(16, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_16_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(16, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_17_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(17, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_17_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(17, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_17_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(17, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_17_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(17, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_17_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(17, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_17_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(17, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_18_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(18, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_18_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(18, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_18_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(18, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_18_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(18, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_18_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(18, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_18_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(18, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_19_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(19, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_19_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(19, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_19_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(19, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_19_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(19, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_19_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(19, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_19_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(19, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_20_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(20, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_20_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(20, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_20_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(20, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_20_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(20, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_20_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(20, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_20_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(20, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_21_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(21, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_21_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(21, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_21_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(21, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_21_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(21, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_21_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(21, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_21_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(21, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_22_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(22, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_22_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(22, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_22_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(22, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_22_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(22, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_22_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(22, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_22_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(22, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_23_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(23, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_23_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(23, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_23_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(23, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_23_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(23, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_23_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(23, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_23_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(23, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


def coco_24_5(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(24, 5, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_24_10(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(24, 10, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_24_20(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(24, 20, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_24_40(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(24, 40, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_24_80(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(24, 80, ia2, ia4, ia8, ia16, ra, ca1, ca2)

def coco_24_160(ia2, ia4, ia8, ia16, ra, ca1, ca2):
    return coco_categorical(24, 160, ia2, ia4, ia8, ia16, ra, ca1, ca2)


if __name__ == '__main__':
    pass
