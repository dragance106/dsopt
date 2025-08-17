"""
Tests on categorical versions of COCO testbed functions,
as well as tests on the optimization of KW villa energy behaviour.
"""
import numpy as np

from dsopt.dsopt import dsopt
import tests_coco_categorical as tc
from kw_villa.kw_villa import simulate_desert_villa

##############################
# DSOPT for COCO categorical #
##############################
def dsopt_coco_aposteriori(f_from=1, f_to=24, n_repeat=8, n_trials=400):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'dsopt_aposteriori: optimizing coco_{f_index}_10, repeat {repeat}')

            fun = getattr(tc, f'coco_{f_index}_10')
            best_value, best_point = dsopt(fun,
                                           {'ia2': (2, [0, 1]),
                                            'ia4': (2, [0, 1, 2, 3]),
                                            'ia8': (2, [0, 1, 2, 3, 4, 5, 6, 7]),
                                            'ia16': (2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                            'ra': (2, np.linspace(-5.0, 5.0, 101)),
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
                                           filename=f'dsopt_aposteriori_coco_{f_index}_repeat_{repeat}.csv')
            print(f'  coco_{f_index}_10, repeat {repeat}: best value={best_value}, best_point={best_point}')


def dsopt_coco_mipt(f_from=1, f_to=24, n_repeat=8, n_trials=400):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'dsopt_mipt: optimizing coco_{f_index}_10, repeat {repeat}')

            fun = getattr(tc, f'coco_{f_index}_10')
            best_value, best_point = dsopt(fun,
                                           {'ia2': (2, [0, 1]),
                                            'ia4': (2, [0, 1, 2, 3]),
                                            'ia8': (2, [0, 1, 2, 3, 4, 5, 6, 7]),
                                            'ia16': (2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                            'ra': (2, np.linspace(-5.0, 5.0, 101)),
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
                                           filename=f'dsopt_mipt_coco_{f_index}_repeat_{repeat}.csv')
            print(f'  coco_{f_index}_10, repeat {repeat}: best value={best_value}, best_point={best_point}')


def dsopt_coco_voronoi(f_from=1, f_to=24, n_repeat=8, n_trials=400):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'dsopt_voronoi: optimizing coco_{f_index}_10, repeat {repeat}')

            fun = getattr(tc, f'coco_{f_index}_10')
            best_value, best_point = dsopt(fun,
                                           {'ia2': (2, [0, 1]),
                                            'ia4': (2, [0, 1, 2, 3]),
                                            'ia8': (2, [0, 1, 2, 3, 4, 5, 6, 7]),
                                            'ia16': (2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                            'ra': (2, np.linspace(-5.0, 5.0, 101)),
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
                                           filename=f'dsopt_voronoi_coco_{f_index}_repeat_{repeat}.csv')
            print(f'  coco_{f_index}_10, repeat {repeat}: best value={best_value}, best_point={best_point}')


######################
# DSOPT for KW villa #
######################
def dsopt_kw_villa_aposteriori(n_repeat=8, n_trials=400):
    for repeat in range(n_repeat):
        print(f'dsopt_aposteriori: optimizing KW villa, repeat {repeat}')

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
                                       filename=f'dsopt_aposteriori_kw_villa_{repeat}.csv')
        print(f'  KW villa, repeat {repeat}: best value={best_value}, best_point={best_point}')


def dsopt_kw_villa_mipt(n_repeat=8, n_trials=400):
    for repeat in range(n_repeat):
        print(f'dsopt_mipt: optimizing KW villa, repeat {repeat}')

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
                                       filename=f'dsopt_mipt_kw_villa_{repeat}.csv')
        print(f'  KW villa, repeat {repeat}: best value={best_value}, best_point={best_point}')


def dsopt_kw_villa_voronoi(n_repeat=8, n_trials=400):
    for repeat in range(n_repeat):
        print(f'dsopt_voronoi: optimizing KW villa, repeat {repeat}')

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
                                       filename=f'dsopt_voronoi_kw_villa_{repeat}.csv')
        print(f'  KW villa, repeat {repeat}: best value={best_value}, best_point={best_point}')


##############################
# SMAC3 for COCO categorical #
##############################
from ConfigSpace import ConfigurationSpace
from smac import Scenario, HyperparameterOptimizationFacade, BlackBoxFacade
import tests_coco_smac_wrapper as coco_smac


def smac_RF_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'smac_RF: optimizing coco_{f_index}_10, repeat {repeat}')

            cs = ConfigurationSpace({'ia2a': (0, 2),
                                     'ia2b': (0, 2),
                                     'ia4a': (0, 4),
                                     'ia4b': (0, 4),
                                     'ia8a': (0, 8),
                                     'ia8b': (0, 8),
                                     'ia16a': (0, 16),
                                     'ia16b': (0, 16),
                                     'raa': (-5.0, 5.0),
                                     'rab': (-5.0, 5.0),
                                     'ca': [0, 1, 2],
                                     'cb': [0, 1, 2, 3, 4]})

            scenario = Scenario(configspace=cs,
                                name=f'smac_RF_coco_{f_index}_10_repeat_{repeat}',
                                n_workers=8,
                                n_trials=n_trials,
                                seed=-1)

            fun = getattr(coco_smac, f'smac_coco_{f_index}_10')
            smac = HyperparameterOptimizationFacade(scenario,
                                                    fun,
                                                    overwrite=True)

            best_point = smac.optimize()
            best_value = smac.validate(best_point)

            # save data about evaluated trials to an external file
            filename = f'smac_RF_coco_{f_index}_10_repeat_{repeat}.csv'
            with open(filename, 'a') as f:
                # header line
                f.write('ia2a, i2ab, ia4a, ia4b, ia8a, ia8b, ia16a, ia16b, raa, rab, ca, cb, smac_value\n')

                # trial data
                for k, v in smac.runhistory.items():
                    config = smac.runhistory.get_config(k.config_id)
                    f.write(f'{config["ia2a"]}, {config["ia2b"]}, {config["ia4a"]}, {config["ia4b"]}, ' + \
                        f'{config["ia8a"]}, {config["ia8b"]}, {config["ia16a"]}, {config["ia16b"]}, ' + \
                        f'{config["raa"]}, {config["rab"]}, {config["ca"]}, {config["cb"]}, {v.cost}\n')

            print(f'  coco_{f_index}_10, repeat {repeat}: best value={best_value}, best_point={best_point}')


def smac_BO_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'smac_BO: optimizing coco_{f_index}_10, repeat {repeat}')

            cs = ConfigurationSpace({'ia2a': (0, 2),
                                     'ia2b': (0, 2),
                                     'ia4a': (0, 4),
                                     'ia4b': (0, 4),
                                     'ia8a': (0, 8),
                                     'ia8b': (0, 8),
                                     'ia16a': (0, 16),
                                     'ia16b': (0, 16),
                                     'raa': (-5.0, 5.0),
                                     'rab': (-5.0, 5.0),
                                     'ca': [0, 1, 2],
                                     'cb': [0, 1, 2, 3, 4]})

            scenario = Scenario(configspace=cs,
                                name=f'smac_BO_coco_{f_index}_10_repeat_{repeat}',
                                n_workers=8,
                                n_trials=n_trials,
                                seed=-1)

            fun = getattr(coco_smac, f'smac_coco_{f_index}_10')
            smac = BlackBoxFacade(scenario,
                                  fun,
                                  overwrite=True)

            best_point = smac.optimize()
            best_value = smac.validate(best_point)

            # save data about evaluated trials to an external file
            filename = f'smac_BO_coco_{f_index}_10_repeat_{repeat}.csv'
            with open(filename, 'a') as f:
                # header line
                f.write('ia2a, i2ab, ia4a, ia4b, ia8a, ia8b, ia16a, ia16b, raa, rab, ca, cb, smac_value\n')

                # trial data
                for k, v in smac.runhistory.items():
                    config = smac.runhistory.get_config(k.config_id)
                    f.write(f'{config["ia2a"]}, {config["ia2b"]}, {config["ia4a"]}, {config["ia4b"]}, ' + \
                        f'{config["ia8a"]}, {config["ia8b"]}, {config["ia16a"]}, {config["ia16b"]}, ' + \
                        f'{config["raa"]}, {config["rab"]}, {config["ca"]}, {config["cb"]}, {v.cost}\n')

            print(f'  coco_{f_index}_10, repeat {repeat}: best value={best_value}, best_point={best_point}')


def smac_RF_kw_villa(n_repeat=8, n_trials=400):
    for repeat in range(n_repeat):
        print(f'smac_RF: optimizing KW villa, repeat {repeat}')

        cs = ConfigurationSpace({'glazing_open_facade': [1, 2, 3, 4, 5, 6],
                                 'shading_open_facade': ['int_shade', 'ext_shade', 'ext_blind'],
                                 'glazing_closed_facade': [1, 2, 3, 4, 5, 6],
                                 'wwr_front': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                               0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                                 'exterior_wall': [1, 2, 3, 4],
                                 'insulation_thickness': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]})

        scenario = Scenario(configspace=cs,
                            name=f'smac_RF_kw_villa_repeat_{repeat}',
                            n_workers=8,
                            n_trials=n_trials,
                            seed=-1)

        smac = HyperparameterOptimizationFacade(scenario,
                                                coco_smac.smac_kw_villa,
                                                overwrite=True)

        best_point = smac.optimize()
        best_value = smac.validate(best_point)

        # save data about evaluated trials to an external file
        filename = f'smac_RF_kw_villa_repeat_{repeat}.csv'
        with open(filename, 'a') as f:
            # header line
            f.write('glazing_open_facade, shading_open_facade, glazing_closed_facade, wwr_front, exterior_wall, insulation_thickness, smac_value\n')

            # trial data
            for k, v in smac.runhistory.items():
                config = smac.runhistory.get_config(k.config_id)
                f.write(f'{config["glazing_open_facade"]}, {config["shading_open_facade"]}, ' + \
                        f'{config["glazing_closed_facade"]}, {config["wwr_front"]}, ' + \
                        f'{config["exterior_wall"]}, {config["insulation_thickness"]}, {v.cost}\n')

        print(f'  kw_villa, repeat {repeat}: best value={best_value}, best_point={best_point}')


def smac_BO_kw_villa(n_repeat=8, n_trials=400):
    for repeat in range(n_repeat):
        print(f'smac_BO: optimizing KW villa, repeat {repeat}')

        cs = ConfigurationSpace({'glazing_open_facade': [1, 2, 3, 4, 5, 6],
                                 'shading_open_facade': ['int_shade', 'ext_shade', 'ext_blind'],
                                 'glazing_closed_facade': [1, 2, 3, 4, 5, 6],
                                 'wwr_front': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                               0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                                 'exterior_wall': [1, 2, 3, 4],
                                 'insulation_thickness': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]})

        scenario = Scenario(configspace=cs,
                            name=f'smac_BO_kw_villa_repeat_{repeat}',
                            n_workers=8,
                            n_trials=n_trials,
                            seed=-1)

        smac = BlackBoxFacade(scenario,
                              coco_smac.smac_kw_villa,
                              overwrite=True)

        best_point = smac.optimize()
        best_value = smac.validate(best_point)

        # save data about evaluated trials to an external file
        filename = f'smac_BO_kw_villa_repeat_{repeat}.csv'
        with open(filename, 'a') as f:
            # header line
            f.write('glazing_open_facade, shading_open_facade, glazing_closed_facade, wwr_front, exterior_wall, insulation_thickness, smac_value\n')

            # trial data
            for k, v in smac.runhistory.items():
                config = smac.runhistory.get_config(k.config_id)
                f.write(f'{config["glazing_open_facade"]}, {config["shading_open_facade"]}, ' + \
                        f'{config["glazing_closed_facade"]}, {config["wwr_front"]}, ' + \
                        f'{config["exterior_wall"]}, {config["insulation_thickness"]}, {v.cost}\n')

        print(f'  kw_villa, repeat {repeat}: best value={best_value}, best_point={best_point}')


##############################
# PYMOO for COCO categorical #
##############################
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.core.variable import Real, Integer, Choice
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize

class MixedVariableProblemCoco(ElementwiseProblem):

    def __init__(self, **kwargs):
        self.f_index = kwargs['f_index']
        vars = {
            'ia2a': Integer(bounds=(0,2)),
            'ia2b': Integer(bounds=(0,2)),
            'ia4a': Integer(bounds=(0, 4)),
            'ia4b': Integer(bounds=(0, 4)),
            'ia8a': Integer(bounds=(0, 8)),
            'ia8b': Integer(bounds=(0, 8)),
            'ia16a': Integer(bounds=(0, 16)),
            'ia16b': Integer(bounds=(0, 16)),
            'raa': Real(bounds=(-5.0, 5.0)),
            'rab': Real(bounds=(-5.0, 5.0)),
            'ca': Choice(options=[0, 1, 2]),
            'cb': Choice(options=[0, 1, 2, 3, 4])
        }
        super().__init__(vars=vars, n_obj=1, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        fun = getattr(tc, f'coco_{self.f_index}_10')
        out["F"] = fun([X['ia2a'], X['ia2b']],
                       [X['ia4a'], X['ia4b']],
                       [X['ia8a'], X['ia8b']],
                       [X['ia16a'], X['ia16b']],
                       [X['raa'], X['rab']],
                       X['ca'], X['cb'])


def pymoo_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'pymoo: optimizing coco_{f_index}_10, repeat {repeat}')

            # PARALLELIZED EXECUTION
            n_threads = 8
            pool = ThreadPool(n_threads)
            runner = StarmapParallelization(pool.starmap)
            problem = MixedVariableProblemCoco(elementwise_runner=runner,
                                               f_index=f_index)     # COCO function index
            algorithm = MixedVariableGA(pop_size=10,
                                        save_history=True)
            res = minimize(problem,
                           algorithm,
                           termination=('n_evals', n_trials),
                           verbose=False)
            pool.close()

            # save data about evaluated individuals over generations to an external file
            filename = f'pymoo_coco_{f_index}_10_repeat_{repeat}.csv'
            with open(filename, 'a') as f:
                # header line
                f.write('ia2a, i2ab, ia4a, ia4b, ia8a, ia8b, ia16a, ia16b, raa, rab, ca, cb, pymoo_value\n')

                # data of evaluated individuals
                for g in res.history:
                    for o in g.pop:
                        f.write(f'{o.X["ia2a"]}, {o.X["ia2b"]}, {o.X["ia4a"]}, {o.X["ia4b"]}, ' + \
                                f'{o.X["ia8a"]}, {o.X["ia8b"]}, {o.X["ia16a"]}, {o.X["ia16b"]}, ' + \
                                f'{o.X["raa"]}, {o.X["rab"]}, {o.X["ca"]}, {o.X["cb"]}, {o.F[0]}\n')

            print(f'  coco_{f_index}_10, repeat {repeat}: best value={res.F}, best_point={res.X}')


######################
# PYMOO for KW villa #
######################
class MixedVariableProblemKWVilla(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {'glazing_open_facade': Choice(options=[1, 2, 3, 4, 5, 6]),
                'shading_open_facade': Choice(options=['int_shade', 'ext_shade', 'ext_blind']),
                'glazing_closed_facade': Choice(options=[1, 2, 3, 4, 5, 6]),
                'wwr_front': Choice(options=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
                'exterior_wall': Choice(options=[1, 2, 3, 4]),
                'insulation_thickness': Choice(options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])}
        super().__init__(vars=vars, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = simulate_desert_villa(X['glazing_open_facade'],
                                         X['shading_open_facade'],
                                         X['glazing_closed_facade'],
                                         X['wwr_front'],
                                         X['exterior_wall'],
                                         X['insulation_thickness'])


def pymoo_kw_villa(n_repeat=8, n_trials=400):
    for repeat in range(n_repeat):
        print(f'pymoo: optimizing KW villa, repeat {repeat}')

        # SINGLE THREADED EXECUTION - STARMAP AND E+ DO NOT LIKE EACH OTHER...
        problem = MixedVariableProblemKWVilla()
        algorithm = MixedVariableGA(pop_size=10,
                                    save_history=True)
        res = minimize(problem,
                       algorithm,
                       termination=('n_evals', n_trials),
                       verbose=False)

        # save data about evaluated individuals over generations to an external file
        filename = f'pymoo_kw_villa_repeat_{repeat}.csv'
        with open(filename, 'a') as f:
            # header line
            f.write('glazing_open_facade, shading_open_facade, glazing_closed_facade, wwr_front, exterior_wall, insulation_thickness, pymoo_value\n')

            # data of evaluated individuals
            for g in res.history:
                for o in g.pop:
                    f.write(f'{o.X["glazing_open_facade"]}, '
                            f'{o.X["shading_open_facade"]}, '
                            f'{o.X["glazing_closed_facade"]}, '
                            f'{o.X["wwr_front"]}, '
                            f'{o.X["exterior_wall"]}, '
                            f'{o.X["insulation_thickness"]}, {o.F[0]}\n')

        print(f'  kw_villa, repeat {repeat}: best value={res.F}, best_point={res.X}')


##############################
# PYSOT for COCO categorical #
##############################
from poap.controller import BasicWorkerThread, ThreadController
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SOPStrategy, DYCORSStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant
from pySOT.optimization_problems import OptimizationProblem


class CocoFunction(OptimizationProblem):
    # Optimization parameters
    hyper_map = {'ia2a': 0, 'ia2b': 1, 'ia4a': 2, 'ia4b': 3,
                 'ia8a': 4, 'ia8b': 5, 'ia16a': 6, 'ia16b': 7,
                 'raa': 8, 'rab': 9, 'ca': 10, 'cb': 11 }

    def __init__(self, **kwargs):
        super().__init__()

        self.dim = 12
        self.lb = np.zeros(self.dim)
        self.ub = np.zeros(self.dim)

        m = self.hyper_map
        self.lb[m['ia2a']] = 0
        self.ub[m['ia2a']] = 1
        self.lb[m['ia2b']] = 0
        self.ub[m['ia2b']] = 1
        self.lb[m['ia4a']] = 0
        self.ub[m['ia4a']] = 3
        self.lb[m['ia4b']] = 0
        self.ub[m['ia4b']] = 3
        self.lb[m['ia8a']] = 0
        self.ub[m['ia8a']] = 7
        self.lb[m['ia8b']] = 0
        self.ub[m['ia8b']] = 7
        self.lb[m['ia16a']] = 0
        self.ub[m['ia16a']] = 15
        self.lb[m['ia16b']] = 0
        self.ub[m['ia16b']] = 15
        self.lb[m['raa']] = -5.0
        self.ub[m['raa']] = 5.0
        self.lb[m['rab']] = -5.0
        self.ub[m['rab']] = 5.0
        self.lb[m['ca']] = 0
        self.ub[m['ca']] = 2
        self.lb[m['cb']] = 0
        self.ub[m['cb']] = 4

        self.cont_var = np.array([m['raa'], m['rab']])
        self.int_var = np.array([m['ia2a'], m['ia2b'], m['ia4a'], m['ia4b'],
                                 m['ia8a'], m['ia8b'], m['ia16a'], m['ia16b'],
                                 m['ca'], m['cb']])

        self.f_index = kwargs['f_index']
        self.repeat = kwargs['repeat']
        self.filename = f'pysot_coco_{self.f_index}_10_repeat_{self.repeat}.csv'

        with open(self.filename, 'a') as f:
            # header line
            f.write('ia2a, i2ab, ia4a, ia4b, ia8a, ia8b, ia16a, ia16b, raa, rab, ca, cb, pysot_value\n')


    def eval(self, x):
        fun = getattr(tc, f'coco_{self.f_index}_10')
        result=fun([int(x[0]), int(x[1])],
                   [int(x[2]), int(x[3])],
                   [int(x[4]), int(x[5])],
                   [int(x[6]), int(x[7])],
                   [x[8], x[9]],
                   int(x[10]), int(x[11]))

        with open(self.filename, 'a') as f:
            # data of each evaluated individual
            f.write(f'{x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}, {x[5]}, {x[6]}, '
                    f'{x[7]}, {x[8]}, {x[9]}, {x[10]}, {x[11]}, {result}\n')

        return result


def pysot_coco(f_from=1, f_to=24, n_repeat=8, n_trials=400):
    for f_index in range(f_from, f_to+1):
        for repeat in range(n_repeat):
            print(f'pysot: optimizing coco_{f_index}_10, repeat {repeat}')

            n_threads = 8
            coco_fun = CocoFunction(f_index=f_index, repeat=repeat)     # PROBLEM DEFINITION!
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
                print(f'  coco_{f_index}_10, repeat {repeat}: best value={result.value}, best_point={result.params[0]}')
            except:
                print(f'  coco_{f_index}_10, repeat {repeat}: PUCE! IDEMO DALJE...')


######################
# PYSOT for KW villa #
######################
class KWVillaFunction(OptimizationProblem):
    # Optimization parameters
    hyper_map = {'glazing_open_facade': 0,
                 'shading_open_facade': 1,
                 'glazing_closed_facade': 2,
                 'wwr_front': 3,
                 'exterior_wall': 4,
                 'insulation_thickness': 5}

    def __init__(self, **kwargs):
        super().__init__()

        self.dim = 6
        self.lb = np.zeros(self.dim)
        self.ub = np.zeros(self.dim)

        m = self.hyper_map
        self.lb[m['glazing_open_facade']] = 1
        self.ub[m['glazing_open_facade']] = 6
        self.lb[m['glazing_closed_facade']] = 1
        self.ub[m['glazing_closed_facade']] = 6
        self.lb[m['exterior_wall']] = 1
        self.ub[m['exterior_wall']] = 4
        self.lb[m['shading_open_facade']] = 0
        self.ub[m['shading_open_facade']] = 2
        self.lb[m['wwr_front']] = 0
        self.ub[m['wwr_front']] = 18
        self.lb[m['insulation_thickness']] = 0
        self.ub[m['insulation_thickness']] = 10

        self.cont_var = np.array([])
        self.int_var = np.array([m['glazing_open_facade'], m['shading_open_facade'],
                                 m['glazing_closed_facade'], m['wwr_front'],
                                 m['exterior_wall'], m['insulation_thickness']])

        self.repeat = kwargs['repeat']
        self.filename = f'pysot_kw_villa_repeat_{self.repeat}.csv'

        with open(self.filename, 'a') as f:
            # header line
            f.write('glazing_open_facade, shading_open_facade, glazing_closed_facade, wwr_front, exterior_wall, insulation_thickness, pysot_value\n')


    def eval(self, x):
        glazing_open_facade = int(x[0])
        shading_options = ['int_shade', 'ext_shade', 'ext_blind']
        shading_open_facade = shading_options[int(x[1])]
        glazing_closed_facade = int(x[2])
        wwr_front_options = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        wwr_front = wwr_front_options[int(x[3])]
        exterior_wall = int(x[4])
        insulation_thickness_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        insulation_thickness = insulation_thickness_options[int(x[5])]

        result = simulate_desert_villa(glazing_open_facade, shading_open_facade, glazing_closed_facade,
                                       wwr_front, exterior_wall, insulation_thickness)

        with open(self.filename, 'a') as f:
            # data of each evaluated individual
            f.write(f'{glazing_open_facade}, {shading_open_facade}, {glazing_closed_facade}, '
                    f'{wwr_front}, {exterior_wall}, {insulation_thickness}, {result}\n')

        return result


def pysot_kw_villa(n_repeat=8, n_trials=400):
    for repeat in range(n_repeat):
        print(f'pysot: optimizing kw_villa, repeat {repeat}')

        kw_villa_fun = KWVillaFunction(repeat=repeat)     # PROBLEM DEFINITION!
        rbf = RBFInterpolant(dim=kw_villa_fun.dim, lb=kw_villa_fun.lb, ub=kw_villa_fun.ub,
                             kernel=CubicKernel(), tail=LinearTail(kw_villa_fun.dim))
        slhd = SymmetricLatinHypercube(dim=kw_villa_fun.dim,
                                       num_pts=2 * (kw_villa_fun.dim + 1))

        # Create a strategy and a controller
        n_threads = 1
        controller = ThreadController()
        # I cannot stop SOP strategy for evaluating the same EnergyPlus design twice,
        # which leads to missing temporary files, errors in execution and inf results...
        # controller.strategy = SOPStrategy(
        # Hence DYCORSStrategy just for general comparison...
        controller.strategy = DYCORSStrategy(
            max_evals=n_trials,
            opt_prob=kw_villa_fun,
            exp_design=slhd,
            surrogate=rbf,
            asynchronous=False,
            # batch_size=n_threads,     # for SOPSStrategy,
            # ncenters=n_threads)       # but alas...
            batch_size=1)

        # Launch the threads and give them access to the objective function
        for _ in range(n_threads):
            worker = BasicWorkerThread(controller, kw_villa_fun.eval)
            controller.launch_worker(worker)

        # Run the optimization strategy
        result = controller.run()

        print(f'  kw_villa repeat {repeat}: best value={result.value}, best_point={result.params[0]}')


##################
# RUN THE TESTS! #
##################
if __name__=="__main__":
    # dsopt_coco_aposteriori()
    # dsopt_coco_mipt()
    # dsopt_coco_voronoi()
    #
    # dsopt_kw_villa_aposteriori()
    # dsopt_kw_villa_mipt()
    # dsopt_kw_villa_voronoi()

    # smac_BO_coco()
    # smac_BO_kw_villa()

    # smac_RF_coco()
    # smac_RF_kw_villa()

    # pymoo_coco()
    # pymoo_kw_villa()

    # pysot_kw_villa()
    # pysot_coco()

    pass

