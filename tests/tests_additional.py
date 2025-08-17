"""
Additional tests on categorical versions of COCO testbed functions,
requested by the reviewer's question: what happens if we raise the dimension?
COCO bbob-mixint offers dimensions 5 (too small), 10 (already used), 20, 40, 80 and 160,
so we deal here with additional dimensions: 20, 40, 80 and 160

Due to high values of these dimensions,
the code below has to be adapted to variable number of dimensions first...

Note that the parameter dim below must belong to [5, 10, 20, 40, 80, 160]!
"""
import numpy as np

from dsopt.dsopt import dsopt
import tests_coco_categorical as tc

##############################
# DSOPT for COCO categorical #
##############################
def dsopt_coco_aposteriori(f_from=1, f_to=24, n_repeat=8, n_trials=400, dim=10):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'dsopt_aposteriori: optimizing coco_{f_index}_{dim}, repeat {repeat}')

            fun = getattr(tc, f'coco_{f_index}_{dim}')
            best_value, best_point = dsopt(fun,
                                           {'ia2': (dim//5, [0, 1]),
                                            'ia4': (dim//5, [0, 1, 2, 3]),
                                            'ia8': (dim//5, [0, 1, 2, 3, 4, 5, 6, 7]),
                                            'ia16': (dim//5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                            'ra': (dim//5, np.linspace(-5.0, 5.0, 101)),
                                            'ca1': ([0, 1, 2], 'c'),
                                            'ca2': ([0, 1, 2, 3, 4], 'c')},
                                           opt='min',
                                           initial_sample_size=32,
                                           iterative_sample_size=8,
                                           max_evaluations=n_trials,
                                           k=100,
                                           uncertainty_metric='aposteriori',
                                           alpha=1.0,
                                           test_set_percentage=0,
                                           omega_percentage=10,
                                           tau_percentage=10,
                                           upsilon_percentage=25,
                                           sigma_t=50,
                                           sigma_u=10,
                                           gamma=0.01,
                                           r=20,
                                           verbose_level=1,
                                           filename=f'dsopt_aposteriori_coco_{f_index}_{dim}_repeat_{repeat}.csv')
            print(f'  coco_{f_index}_{dim}, repeat {repeat}: best value={best_value}, best_point={best_point}')


def dsopt_coco_mipt(f_from=1, f_to=24, n_repeat=8, n_trials=400, dim=10):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'dsopt_mipt: optimizing coco_{f_index}_{dim}, repeat {repeat}')

            fun = getattr(tc, f'coco_{f_index}_{dim}')
            best_value, best_point = dsopt(fun,
                                           {'ia2': (dim//5, [0, 1]),
                                            'ia4': (dim//5, [0, 1, 2, 3]),
                                            'ia8': (dim//5, [0, 1, 2, 3, 4, 5, 6, 7]),
                                            'ia16': (dim//5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                            'ra': (dim//5, np.linspace(-5.0, 5.0, 101)),
                                            'ca1': ([0, 1, 2], 'c'),
                                            'ca2': ([0, 1, 2, 3, 4], 'c')},
                                           opt='min',
                                           initial_sample_size=32,
                                           iterative_sample_size=8,
                                           max_evaluations=n_trials,
                                           k=100,
                                           uncertainty_metric='mipt',
                                           alpha=1.0,
                                           test_set_percentage=0,
                                           omega_percentage=10,
                                           tau_percentage=10,
                                           upsilon_percentage=25,
                                           sigma_t=50,
                                           sigma_u=10,
                                           gamma=0.01,
                                           r=20,
                                           verbose_level=1,
                                           filename=f'dsopt_mipt_coco_{f_index}_{dim}_repeat_{repeat}.csv')
            print(f'  coco_{f_index}_{dim}, repeat {repeat}: best value={best_value}, best_point={best_point}')


def dsopt_coco_voronoi(f_from=1, f_to=24, n_repeat=8, n_trials=400, dim=10):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'dsopt_voronoi: optimizing coco_{f_index}_{dim}, repeat {repeat}')

            fun = getattr(tc, f'coco_{f_index}_{dim}')
            best_value, best_point = dsopt(fun,
                                           {'ia2': (dim//5, [0, 1]),
                                            'ia4': (dim//5, [0, 1, 2, 3]),
                                            'ia8': (dim//5, [0, 1, 2, 3, 4, 5, 6, 7]),
                                            'ia16': (dim//5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                            'ra': (dim//5, np.linspace(-5.0, 5.0, 101)),
                                            'ca1': ([0, 1, 2], 'c'),
                                            'ca2': ([0, 1, 2, 3, 4], 'c')},
                                           opt='min',
                                           initial_sample_size=32,
                                           iterative_sample_size=8,
                                           max_evaluations=n_trials,
                                           k=100,
                                           uncertainty_metric='voronoi',
                                           alpha=1.0,
                                           test_set_percentage=0,
                                           omega_percentage=10,
                                           tau_percentage=10,
                                           upsilon_percentage=25,
                                           sigma_t=50,
                                           sigma_u=10,
                                           gamma=0.01,
                                           r=20,
                                           verbose_level=1,
                                           filename=f'dsopt_voronoi_coco_{f_index}_{dim}_repeat_{repeat}.csv')
            print(f'  coco_{f_index}_{dim}, repeat {repeat}: best value={best_value}, best_point={best_point}')


##############################
# SMAC3 for COCO categorical #
##############################
from ConfigSpace import ConfigurationSpace
from smac import Scenario, HyperparameterOptimizationFacade, BlackBoxFacade
import tests_coco_smac_wrapper as coco_smac


def smac_RF_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400, dim=10):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'smac_RF: optimizing coco_{f_index}_{dim}, repeat {repeat}')

            arg_dict = {}
            for i in range(dim//5):
                arg_dict[f'ia2{i}'] = (0, 2)
            for i in range(dim//5):
                arg_dict[f'ia4{i}'] = (0, 4)
            for i in range(dim//5):
                arg_dict[f'ia8{i}'] = (0, 8)
            for i in range(dim//5):
                arg_dict[f'ia16{i}'] = (0, 16)
            for i in range(dim//5):
                arg_dict[f'ra{i}'] = (-5.0, 5.0)
            arg_dict['ca'] = [0, 1, 2]
            arg_dict['cb'] = [0, 1, 2, 3, 4]

            cs = ConfigurationSpace(arg_dict)

            scenario = Scenario(configspace=cs,
                                name=f'smac_RF_coco_{f_index}_{dim}_repeat_{repeat}',
                                n_workers=8,
                                n_trials=n_trials,
                                seed=-1)

            fun = getattr(coco_smac, f'smac_coco_{f_index}_{dim}')
            smac = HyperparameterOptimizationFacade(scenario,
                                                    fun,
                                                    overwrite=True)

            best_point = smac.optimize()
            best_value = smac.validate(best_point)

            # save data about evaluated trials to an external file
            filename = f'smac_RF_coco_{f_index}_{dim}_repeat_{repeat}.csv'
            with open(filename, 'a') as f:
                # header line
                print(f'{str(list(arg_dict.keys()))[1:-1]}, smac_value', file=f)
                # f.write('ia2a, i2ab, ia4a, ia4b, ia8a, ia8b, ia16a, ia16b, raa, rab, ca, cb, smac_value\n')

                # trial data
                for k, v in smac.runhistory.items():
                    config = smac.runhistory.get_config(k.config_id)

                    # config.values do not appear in the same order as in arg_dict!
                    s = [config[f'ia2{i}'] for i in range(dim // 5)] + \
                        [config[f'ia4{i}'] for i in range(dim // 5)] + \
                        [config[f'ia8{i}'] for i in range(dim // 5)] + \
                        [config[f'ia16{i}'] for i in range(dim // 5)] + \
                        [config[f'ra{i}'] for i in range(dim // 5)] + \
                        [config['ca'], config['cb'], v.cost]
                    print(f'{str(s)[1:-1]}', file=f)

                    # f.write(f'{config["ia2a"]}, {config["ia2b"]}, {config["ia4a"]}, {config["ia4b"]}, ' + \
                    #     f'{config["ia8a"]}, {config["ia8b"]}, {config["ia16a"]}, {config["ia16b"]}, ' + \
                    #     f'{config["raa"]}, {config["rab"]}, {config["ca"]}, {config["cb"]}, {v.cost}\n')

            print(f'  coco_{f_index}_{dim}, repeat {repeat}: best value={best_value}, best_point={best_point}')


def smac_BO_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400, dim=10):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'smac_BO: optimizing coco_{f_index}_{dim}, repeat {repeat}')

            arg_dict = {}
            for i in range(dim//5):
                arg_dict[f'ia2{i}'] = (0, 2)
            for i in range(dim//5):
                arg_dict[f'ia4{i}'] = (0, 4)
            for i in range(dim//5):
                arg_dict[f'ia8{i}'] = (0, 8)
            for i in range(dim//5):
                arg_dict[f'ia16{i}'] = (0, 16)
            for i in range(dim//5):
                arg_dict[f'ra{i}'] = (-5.0, 5.0)
            arg_dict['ca'] = [0, 1, 2]
            arg_dict['cb'] = [0, 1, 2, 3, 4]

            cs = ConfigurationSpace(arg_dict)

            # cs = ConfigurationSpace({'ia2a': (0, 2),
            #                          'ia2b': (0, 2),
            #                          'ia4a': (0, 4),
            #                          'ia4b': (0, 4),
            #                          'ia8a': (0, 8),
            #                          'ia8b': (0, 8),
            #                          'ia16a': (0, 16),
            #                          'ia16b': (0, 16),
            #                          'raa': (-5.0, 5.0),
            #                          'rab': (-5.0, 5.0),
            #                          'ca': [0, 1, 2],
            #                          'cb': [0, 1, 2, 3, 4]})

            scenario = Scenario(configspace=cs,
                                name=f'smac_BO_coco_{f_index}_{dim}_repeat_{repeat}',
                                n_workers=8,
                                n_trials=n_trials,
                                seed=-1)

            fun = getattr(coco_smac, f'smac_coco_{f_index}_{dim}')
            smac = BlackBoxFacade(scenario,
                                  fun,
                                  overwrite=True)

            best_point = smac.optimize()
            best_value = smac.validate(best_point)

            # save data about evaluated trials to an external file
            filename = f'smac_BO_coco_{f_index}_{dim}_repeat_{repeat}.csv'
            with open(filename, 'a') as f:
                # header line
                print(f'{", ".join(arg_dict.keys())}, smac_value', file=f)
                # f.write('ia2a, i2ab, ia4a, ia4b, ia8a, ia8b, ia16a, ia16b, raa, rab, ca, cb, smac_value\n')

                # trial data
                for k, v in smac.runhistory.items():
                    config = smac.runhistory.get_config(k.config_id)

                    # config.values do not appear in the same order as in arg_dict!
                    s = [config[f'ia2{i}'] for i in range(dim // 5)] + \
                        [config[f'ia4{i}'] for i in range(dim // 5)] + \
                        [config[f'ia8{i}'] for i in range(dim // 5)] + \
                        [config[f'ia16{i}'] for i in range(dim // 5)] + \
                        [config[f'ra{i}'] for i in range(dim // 5)] + \
                        [config['ca'], config['cb'], v.cost]
                    print(f'{str(s)[1:-1]}', file=f)

                    # f.write(f'{config["ia2a"]}, {config["ia2b"]}, {config["ia4a"]}, {config["ia4b"]}, ' + \
                    #     f'{config["ia8a"]}, {config["ia8b"]}, {config["ia16a"]}, {config["ia16b"]}, ' + \
                    #     f'{config["raa"]}, {config["rab"]}, {config["ca"]}, {config["cb"]}, {v.cost}\n')

            print(f'  coco_{f_index}_{dim}, repeat {repeat}: best value={best_value}, best_point={best_point}')


##############################
# PYMOO for COCO categorical #
##############################
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.core.variable import Real, Integer, Choice
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import tests_coco_pymoo_wrapper as coco_pymoo

class MixedVariableProblemCoco(ElementwiseProblem):

    def __init__(self, **kwargs):
        self.f_index = kwargs['f_index']
        self.dim = kwargs['dim']

        vars = {}
        for i in range(dim // 5):
            vars[f'ia2{i}'] = Integer(bounds=(0, 2))
        for i in range(dim // 5):
            vars[f'ia4{i}'] = Integer(bounds=(0, 4))
        for i in range(dim // 5):
            vars[f'ia8{i}'] = Integer(bounds=(0, 8))
        for i in range(dim // 5):
            vars[f'ia16{i}'] = Integer(bounds=(0, 16))
        for i in range(dim // 5):
            vars[f'ra{i}'] = Real(bounds=(-5.0, 5.0))
        vars['ca'] = Choice(options=[0, 1, 2])
        vars['cb'] = Choice(options=[0, 1, 2, 3, 4])

        # vars = {
        #     'ia2a': Integer(bounds=(0,2)),
        #     'ia2b': Integer(bounds=(0,2)),
        #     'ia4a': Integer(bounds=(0, 4)),
        #     'ia4b': Integer(bounds=(0, 4)),
        #     'ia8a': Integer(bounds=(0, 8)),
        #     'ia8b': Integer(bounds=(0, 8)),
        #     'ia16a': Integer(bounds=(0, 16)),
        #     'ia16b': Integer(bounds=(0, 16)),
        #     'raa': Real(bounds=(-5.0, 5.0)),
        #     'rab': Real(bounds=(-5.0, 5.0)),
        #     'ca': Choice(options=[0, 1, 2]),
        #     'cb': Choice(options=[0, 1, 2, 3, 4])
        # }
        super().__init__(vars=vars, n_obj=1, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        fun = getattr(coco_pymoo, f'pymoo_coco_{self.f_index}_{self.dim}')
        out["F"] = fun(X)


def pymoo_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400, dim=10):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'pymoo: optimizing coco_{f_index}_{dim}, repeat {repeat}')

            # PARALLELIZED EXECUTION
            n_threads = 8
            pool = ThreadPool(n_threads)
            runner = StarmapParallelization(pool.starmap)
            problem = MixedVariableProblemCoco(elementwise_runner=runner,
                                               f_index=f_index,     # COCO function index
                                               dim=dim)             # problem dimension
            algorithm = MixedVariableGA(pop_size=10,
                                        save_history=True)
            res = minimize(problem,
                           algorithm,
                           termination=('n_evals', n_trials),
                           verbose=False)
            pool.close()

            # save data about evaluated individuals over generations to an external file
            filename = f'pymoo_coco_{f_index}_{dim}_repeat_{repeat}.csv'
            with open(filename, 'a') as f:
                # header line
                s = [f'ia2{i}' for i in range(dim//5)] + \
                    [f'ia4{i}' for i in range(dim//5)] + \
                    [f'ia8{i}' for i in range(dim//5)] + \
                    [f'ia16{i}' for i in range(dim//5)] + \
                    [f'ra{i}' for i in range(dim//5)] + \
                    ['ca', 'cb', 'pymoo_value']
                print(', '.join(s), file=f)

                # data of evaluated individuals
                for g in res.history:
                    for o in g.pop:
                        print(f'{str(list(o.X.values()))[1:-1]}, {o.F[0]}', file=f)
                        # f.write(f'{o.X["ia2a"]}, {o.X["ia2b"]}, {o.X["ia4a"]}, {o.X["ia4b"]}, ' + \
                        #         f'{o.X["ia8a"]}, {o.X["ia8b"]}, {o.X["ia16a"]}, {o.X["ia16b"]}, ' + \
                        #         f'{o.X["raa"]}, {o.X["rab"]}, {o.X["ca"]}, {o.X["cb"]}, {o.F[0]}\n')

            print(f'  coco_{f_index}_{dim}, repeat {repeat}: best value={res.F}, best_point={res.X}')


##############################
# PYSOT for COCO categorical #
##############################
from poap.controller import BasicWorkerThread, ThreadController
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SOPStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant
from pySOT.optimization_problems import OptimizationProblem
import tests_coco_pysot_wrapper as coco_pysot


class CocoFunction(OptimizationProblem):
    def __init__(self, **kwargs):
        super().__init__()

        self.dim = kwargs['dim'] + 2    # for two additional categorical parameters ca and cb
        self.lb = np.zeros(self.dim)
        self.ub = np.zeros(self.dim)

        fifth = kwargs['dim']//5
        for i in range(fifth):
            self.lb[i] = 0
            self.ub[i] = 1
        for i in range(fifth, 2*fifth):
            self.lb[i] = 0
            self.ub[i] = 3
        for i in range(2*fifth, 3*fifth):
            self.lb[i] = 0
            self.ub[i] = 7
        for i in range(3*fifth, 4*fifth):
            self.lb[i] = 0
            self.ub[i] = 15
        for i in range(4*fifth, 5*fifth):
            self.lb[i] = -5.0
            self.ub[i] = 5.0
        # the last two categorical parameters
        self.lb[5*fifth] = 0
        self.ub[5*fifth] = 2
        self.lb[5*fifth+1] = 0
        self.ub[5*fifth+1] = 4

        self.cont_var = np.array(list(range(4*fifth, 5*fifth)))
        self.int_var = np.array(list(range(0, 4*fifth)) + [5*fifth, 5*fifth+1])

        self.f_index = kwargs['f_index']
        self.repeat = kwargs['repeat']
        self.filename = f'pysot_coco_{self.f_index}_{kwargs["dim"]}_repeat_{self.repeat}.csv'

        with open(self.filename, 'a') as f:
            # header line
            s = [f'ia2{i}' for i in range(fifth)] + \
                [f'ia4{i}' for i in range(fifth)] + \
                [f'ia8{i}' for i in range(fifth)] + \
                [f'ia16{i}' for i in range(fifth)] + \
                [f'ra{i}' for i in range(fifth)] + \
                ['ca', 'cb', 'pysot_value']
            print(', '.join(s), file=f)


    def eval(self, x):
        # fun = getattr(tc, f'coco_{self.f_index}_{dim}')
        # result=fun([int(x[0]), int(x[1])],
        #            [int(x[2]), int(x[3])],
        #            [int(x[4]), int(x[5])],
        #            [int(x[6]), int(x[7])],
        #            [x[8], x[9]],
        #            int(x[10]), int(x[11]))

        fun = getattr(coco_pysot, f'pysot_coco_{self.f_index}_{self.dim-2}')
        result = fun(x)

        with open(self.filename, 'a') as f:
            # data of each evaluated individual
            print(f'{str(x)[1:-1]}, {result}', file=f)
            # f.write(f'{x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}, {x[5]}, {x[6]}, '
            #         f'{x[7]}, {x[8]}, {x[9]}, {x[10]}, {x[11]}, {result}\n')

        return result


def pysot_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400, dim=10):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'pysot: optimizing coco_{f_index}_{dim}, repeat {repeat}')

            n_threads = 8
            coco_fun = CocoFunction(f_index=f_index, dim=dim, repeat=repeat)     # PROBLEM DEFINITION!
            rbf = RBFInterpolant(dim=coco_fun.dim, lb=coco_fun.lb, ub=coco_fun.ub,
                                 kernel=CubicKernel(), tail=LinearTail(coco_fun.dim))
            slhd = SymmetricLatinHypercube(dim=coco_fun.dim,
                                           num_pts=2 * (coco_fun.dim + 1))

            # Create a strategy and a controller
            controller = ThreadController()
            controller.strategy = SOPStrategy(
                max_evals=n_trials,
                opt_prob=coco_fun,
                exp_design=slhd,
                surrogate=rbf,
                asynchronous=False,
                batch_size=n_threads,
                ncenters=n_threads)

            # Launch the threads and give them access to the objective function
            for _ in range(n_threads):
                worker = BasicWorkerThread(controller, coco_fun.eval)
                controller.launch_worker(worker)

            # Run the optimization strategy
            try:
                result = controller.run()
                print(f'  coco_{f_index}_{dim}, repeat {repeat}: best value={result.value}, best_point={result.params[0]}')
            except:
                print(f'  coco_{f_index}_{dim}, repeat {repeat}: PUCE! IDEMO DALJE...')


##################
# RUN THE TESTS! #
##################
if __name__=="__main__":
    # first try only the functions that are illustrated in the original manuscript: F_13, F_15, F_18, F_21, F_22, F_24

    # for dim in [20, 40, 80]:
    #     for f_index in [13, 15, 18, 21, 22, 24]:
    #         dsopt_coco_mipt(f_from=f_index, f_to=f_index, dim=dim)
    #         dsopt_coco_voronoi(f_from=f_index, f_to=f_index, dim=dim)
    #         dsopt_coco_aposteriori(f_from=f_index, f_to=f_index, dim=dim)

    # for dim in [20, 40, 80]:
    #     for f_index in [13, 15, 18, 21, 22, 24]:
    #         pymoo_coco(f_from=f_index, f_to=f_index, dim=dim)
    #         pysot_coco(f_from=f_index, f_to=f_index, dim=dim)


    # for dim in [20, 40, 80]:
    #     for f_index in [13, 15, 18, 21, 22, 24]:
    #         smac_BO_coco(f_from=f_index, f_to=f_index, dim=dim)
    #         smac_RF_coco(f_from=f_index, f_to=f_index, dim=dim)

    for dim in [20, 40, 80]:
        for f_index in [13, 15, 18, 21, 22, 24]:
            pysot_coco(f_from=f_index, f_to=f_index, dim=dim)

    pass

