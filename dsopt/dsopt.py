"""
   Implementation of discrete surrogate optimization
   that relies on Pareto solutions to solve the exploitation-exploration conundrum.

   Possible improvements over other proposed surrogate optimization algorithms are:
   - the use of Pareto solutions instead of
     the optimal value of a linear combination of exploitation and exploration criteria
   - the use of MIPT combination of intersite and projected distances
     instead of simply intersite distance as in entmoot
   - the use of a posteriori uncertainty over distances in the search space
   - the use of prediction of Voronoi cell volumes over distances in the search space

   It is assumed that all the arguments of the expensive function are discrete,
   either numerical or categorical.
   In case some of them are continuous, their range should be replaced beforehand
   by a discrete sample selected at an appropriate resolution.

   It is assumed that the expensive function returns a single real value, which should be optimized.
   The opt parameter will determine whether the goal is to minimize or to maximize this function.
"""
import numpy as np
import xgboost as xgb
import multiprocessing as mp
import time


###########################################
# Useful methods from other package files #
###########################################
from dsopt.arguments import analyze_arguments, hypercube_to_arguments, hypercube_to_sample
from dsopt.mipt import mipt
from dsopt.pareto import select_candidates
from dsopt.uncertainty import MIPTUncertainty, APosterioriUncertainty, VoronoiUncertainty


#############################################################
# An auxiliary method used in parallel computation          #
# to evaluate the expensive function at a given point       #
#############################################################
def compute_exp_func(args):
    exp_func = args[0]
    point = args[1]
    return exp_func(**point)


##########################################################################
# The main discrete surrogate optimization method and its many arguments #
##########################################################################
def dsopt(exp_func,
          arg_dict,
          opt='min',
          initial_sample_size=50,
          iterative_sample_size=8,
          max_evaluations=1000,
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
          r=10,
          verbose_level=0,
          seed=None,
          filename=None):
    """
    A unified framework for Monte Carlo discrete/categorical surrogate optimization,
    where after the initial training,
    in each next iteration
    one selects a large number of random candidate solutions,
    predicts the expensive function value and the uncertainty metric for each of them,
    and then selects a mix of candidates for further expensive function evaluations:
    - among low uncertainty candidates, those with the best predictions,
    - among medium uncertainty candidates,
      Pareto solutions with increasing uncertainties and increasing predictions
      selected uniformly at random, and
    - among high uncertainty candidates, those with the highest uncertainties.

    The uncertainty can be predicted either through:
    - Crombecq's MIPT approach that combines Euclidean and projected distance
      between new candidate points and already computed points,
    - predicting errors between predicted and computed values for new candidate points
      based on historical relative errors for already computed values,
      exponentially decreased by the factor alpha (that indicates higher certainty of
      more recent surrogate models in the vicinity of points computed longer ago)
    - stochastically estimating the volume of Voronoi cells around already computed points
      by enumerating closest candidate points, training a new ML model on these estimates and
      then using predictions of this model at new candidate points as their uncertainties

    The surrogate model employed to predict the values of the expensive function is XGBoost.
    Training can either be done using all computed values so far as the training set,
    or by setting aside a percentage of computed values as the test set.

    :param exp_func:
            the name of the expensive function to be called
    :param arg_dict:
            the dictionary of the arguments of the expensive function.
            Keys represent the argument names.
            Values represent feasible values of the arguments,
            however:
            - if the argument is a numerical scalar,
              then the value is an array containing feasible values
            - if the argument is a numerical array,
              then the value is a tuple (K, array),
              where K denotes the number of times the argument is repeated,
              so that in the expanded form,
              the arguments are 'key0', 'key1', ..., 'keyK-1'
            - if the argument is a categorical scalar,
              then the value is a tuple (array, 'c')
              [as a matter of fact, if the first element of the tuple is an array,
               anything will work instead of 'c' :]
            - if the argument is a categorical array,
              then the value is a tuple (K, array, 'c'),
              where K denotes the number of times the argument is repeated,
              so that in the expanded form,
              the arguments are 'key0', 'key1', ..., 'keyK-1'
            Example of a scalar argument:
              'x': np.linspace(-10, 10, 500)
            Example of a categorical argument
              'x': (['left', 'right'], 'c')
            Example of an array argument:
              'xx': (10, np.linspace(-32.768, 32.768, 100))
            Example of a categorical array argument:
              'xx': (10, ['left', 'right'], 'c')
    :param opt:
            whether to minimize ('min') or maximize ('max') the expensive function
    :param initial_sample_size:
            the size of the initial sample.
            The expensive function will be computed for each of these points,
            and the XGBoost surrogate model initially trained on its values.
    :param iterative_sample_size:
            the size of the iterative sample.
            In each iteration, the method will determine AT MOST this many new points,
            evaluate expensive function for them,
            and update the XGBoost surrogate model using all values computed so far.
            Since the iterative sample consists of the Pareto front
            for a large set of new candidate points,
            there is no guarantee that each iteration will produce iterative
    :param max_evaluations:
            the maximum number of evaluations of the expensive function
            before the surrogate optimization is terminated
    :param k:
            the constant multiplier indicating how many more new candidates
            should be generated when compared to the number of existing points
    :param uncertainty_metric:
            the metric used to predict uncertainty of new candidate points,
            either 'mipt' for Crombecq's MIPT-based mix of Euclidean and projected distances
            or 'aposteriori' for predictions based on historical relative errors
            or 'voronoi' for predictions based on estimates of Voronoi cell volumes
    :param alpha:
            the factor exponentially multiplying older relative errors
            between the simulated and the predicted values
            (used only when uncertainty_metric='aposteriori')
    :param test_set_percentage:
            the percentage of computed values to set aside as the test set.
            If left at the default value of 0,
            then all computed values will be used as the training set for the surrogate model
    :param omega_percentage:
            when selecting candidates for the new sample,
            the allowed percentage of lagging behind best_simulated_value
    :param tau_percentage:
            when selecting candidates for the new sample,
            this percentage of lowest uncertainties determine low uncertainty candidates
    :param upsilon_percentage:
            when selecting candidates for the new sample,
            this percentage of highest uncertainties determine high uncertainty candidates
    :param sigma_t:
            when selecting candidates for the new sample,
            percentage of selected candidates with low uncertainty
    :param sigma_u:
            when selecting candidates for the new sample,
            percentage of selected candidates with high uncertainty
    :param gamma:
            surrogate optimization will be restarted from scratch
            if the best simulated value does not improve at least gamma% over the last r iterations
    :param r:
            the number of iterations without sufficient improvement
            after which surrogate optimization should be restarted from scratch
    :param verbose_level:
            determines the amount of information printed during optimization
            (0=none, 1=csv printout of simulated points only, 2=report on each sample, 3=timings)
    :param seed:
            seed for initializing random number generator,
            enables reproducibility of results if set to a concrete value
    :param filename:
            name of the csv file for reporting the progress of optimization, provided verbose_level>=1
            (sampled designs, as well as the best simulated design at each step)

    :return: the optimal value of the expensive function that has been found so far and
             the values of arguments for which this value has been found
    """

    ###############################################
    # for printout of simulated points to csv file
    ###############################################
    if filename is None:
        filename = 'report_' + exp_func.__name__ \
                   + '_' + opt \
                   + '_eval_' + str(max_evaluations) \
                   + '_init_' + str(initial_sample_size) \
                   + '_iter_' + str(iterative_sample_size) \
                   + '_' + uncertainty_metric \
                   + '_omega_' + str(omega_percentage) \
                   + '_tau_' + str(tau_percentage) \
                   + '_upsilon_' + str(upsilon_percentage) \
                   + '_sigma_t_' + str(sigma_t) \
                   + '_sigma_u_' + str(sigma_u) \
                   + '_gamma_' + str(gamma) \
                   + '_r_' + str(r) + '.csv'

    ########################################################################
    # initialize random number generator with a concrete seed (if provided)
    ########################################################################
    rng = np.random.default_rng(seed)

    ###########################################################
    # pick up from the argument dictionary the information on:
    # - the total dimension of arguments of the expensive function,
    # - the list indicating whether the expanded arguments are numerical or categorical,
    # - the arrays containing feasible values for each argument
    ############################################################
    dim, arg_type, arg_array, arg_len = analyze_arguments(arg_dict)

    #####################################################################
    # restart strategy: the whole process starts from scratch
    # if no sufficient improvement was seen during the last r iterations
    #####################################################################
    iteration = 0
    total_evaluated = 0

    #######################
    # let the games begin!
    #######################
    total_best_value = None                 # "ordinary" best_value and best_point
    total_best_point = None                 # refer only to the current restart...

    # outer while loop
    while total_evaluated < max_evaluations:

        #######################################
        # create the initial sample using MIPT
        # for the moment, we will just assume that
        # total_evaluated + initial_sample_size < max_evaluations :)
        #######################################
        tic = time.perf_counter()
        computed_hyper = mipt(initial_sample_size, dim, rng)
        computed_sample = hypercube_to_sample(computed_hyper, dim, arg_type, arg_array, arg_len)
        # hyper returned by mipt may point to same points, which we do not want to evaluate twice!
        computed_sample = computed_sample.drop_duplicates()
        keep_indices = computed_sample.index.values
        computed_hyper = computed_hyper[keep_indices]
        computed_points = hypercube_to_arguments(computed_hyper, arg_dict)

        toc = time.perf_counter()           # time-reporting?
        if verbose_level >= 3:
            print(f'  timing: initial sample created in {toc - tic:.4f} seconds')
        tic = toc

        ###################################################################
        # compute the expensive function on the initial sample in parallel
        ###################################################################
        material = [(exp_func, point) for point in computed_points]
        with mp.Pool(max(mp.cpu_count(), 1)) as pool:
            computed_values = pool.map(compute_exp_func, material)

        evaluated = len(computed_values)                # evaluated points after the last restart
        total_evaluated += len(computed_values)         # there may have been a few restarts in the meantime...

        toc = time.perf_counter()           # time-reporting?
        if verbose_level >= 3:
            print(f'  timing: initial sample evaluated in {toc - tic:.4f} seconds')
        tic = toc

        ######################
        # initial bookkeeping
        ######################
        if opt == 'max':
            opt_func = np.argmax
        else:
            opt_func = np.argmin

        index = opt_func(computed_values)
        best_sample = computed_sample.iloc[index]
        best_point = computed_points[index]
        best_value = computed_values[index]

        # perhaps we moved the total best value as well?
        if opt == 'max':
            if total_best_value is None or total_best_value < best_value:
                total_best_value = best_value
                total_best_point = best_point
        else: # opt == 'min'
            if total_best_value is None or total_best_value > best_value:
                total_best_value = best_value
                total_best_point = best_point

        ################################################################
        # restart strategy: evaluating the initial sample is considered
        # an improvement over the current/previous best value
        ################################################################
        iter_last_improvement = iteration
        previous_best_value = best_value

        if verbose_level >= 2:                  # report on the initial sample?
            print(f'evaluated the initial sample of {evaluated} points (total evaluated: {total_evaluated})')
            print(f'  best value={best_value} for point: {best_point}')

        #############################################################################
        # train XGBoost surrogate model for initial values of the expensive function
        #############################################################################
        xgb_model = xgb.XGBRegressor(tree_method='approx',
                                     enable_categorical=True,
                                     max_cat_to_onehot=1,
                                     learning_rate=0.1,
                                     verbosity=0)

        # do we use all data for training or do we split it into training and test set?
        if test_set_percentage < 1:
            xgb_model.fit(computed_sample, computed_values, verbose=False)
        else:
            rnd_indices = rng.permutation(evaluated)
            train_size = int(evaluated * (100-test_set_percentage) / 100)

            # computed_sample is a pandas dataframe,
            # computed_values is a list

            train_sample = computed_sample.iloc[rnd_indices[0:train_size], :]
            train_values = [computed_values[i] for i in rnd_indices[0:train_size]]

            test_sample = computed_sample.iloc[rnd_indices[train_size:evaluated], :]
            test_values = [computed_values[i] for i in rnd_indices[train_size:evaluated]]

            xgb_model.fit(train_sample, train_values,
                          eval_set=[(test_sample, test_values)],
                          verbose=False)

        toc = time.perf_counter()               # time-reporting?
        if verbose_level >= 3:
            print(f'  timing: initial surrogate model trained in {toc - tic:.4f} seconds')
        tic = toc

        #################################################
        # set up uncertainty model on the initial sample
        #################################################
        if uncertainty_metric == 'mipt':
            uncertainty_model = MIPTUncertainty(dim)
        elif uncertainty_metric == 'aposteriori':
            uncertainty_model = APosterioriUncertainty(computed_sample, computed_values, alpha)
        else:       # uncertainty_metric == 'voronoi'
            uncertainty_model = VoronoiUncertainty()

        toc = time.perf_counter()               # time-reporting?
        if verbose_level >= 3:
            print(f'  timing: initial uncertainty estimated in {toc - tic:.4f} seconds')
        tic = toc

        ###############################################
        # csv file reporting - if verbose_level >= 1! #
        ###############################################
        if verbose_level >= 1:
            with open(filename, 'a') as f:
                # header line
                line = 'iteration'                  # iteration number
                for i in range(dim):                # generic argument names
                    line += f', arg{i}'
                line += ', computed'                # computed value
                line += ', exploitation'            # exploitation metric (usually equal to computed value)
                line += ', uncertainty'             # exploration metric
                line += ', best value'              # best computed value seen so far
                for i in range(dim):
                    line += f', best_arg{i}'         # generic argument names for best point
                f.write(line+'\n')

                # now for each computed point
                for i in range(evaluated):
                    # iteration number
                    line = str(iteration)

                    # argument values for every computed point
                    for j in range(dim):
                        line += f', {computed_sample.iat[i,j]}'

                    # computed function value for each point
                    line += f', {computed_values[i]}'

                    # exploitation and exploration are zeros at the beginning
                    line += ', 0, 0'

                    # best value seen so far
                    line += f', {best_value}'

                    # argument values for best point so far
                    for j in range(dim):
                        line += f', {best_sample.iat[j]}'

                    f.write(line+'\n')

        ###################################################################
        # run iterations as long as we have available evaluation resources
        # inner while loop
        ###################################################################
        while total_evaluated < max_evaluations:
            iteration += 1

            # how many function evaluations remain?
            max_new_points = min(max_evaluations - total_evaluated, iterative_sample_size)

            ################################################
            # create a large sample of new candidate points
            ################################################
            how_many = k * (evaluated + iterative_sample_size)
            cand_hyper = rng.random((how_many, dim))

            #######################################################################
            # predict values for the expensive function and the uncertainty metric
            #######################################################################
            cand_sample = hypercube_to_sample(cand_hyper, dim, arg_type, arg_array, arg_len)
            cand_points = hypercube_to_arguments(cand_hyper, arg_dict)

            # we do not want duplicate samples among candidates
            cand_sample = cand_sample.drop_duplicates()
            keep_indices = cand_sample.index.values

            # we also do not want candidates that have been already evaluated
            keep_indices = [i for i in keep_indices if cand_points[i] not in computed_points]

            cand_hyper = cand_hyper[keep_indices]
            cand_points = [cand_points[i] for i in keep_indices]
            cand_sample = cand_sample.loc[keep_indices]

            # predict values
            cand_values = xgb_model.predict(cand_sample)
            # predict uncertainties
            if uncertainty_metric == 'mipt':
                cand_expl = uncertainty_model.predict(cand_hyper, computed_hyper)
            elif uncertainty_metric == 'aposteriori':
                cand_expl = uncertainty_model.predict(cand_sample)
            else:  # uncertainty_metric == 'voronoi'
                cand_expl = uncertainty_model.predict(cand_hyper, computed_hyper)

            toc = time.perf_counter()               # time-reporting?
            if verbose_level >= 3:
                print(f'  timing: values and uncertainties for candidate sample predicted in {toc - tic:.4f} seconds')
            tic = toc

            ###############################################################################
            # select a combination of the candidates with best predicted values
            # and the uniformly distributed Pareto solutions with smaller predicted values
            # for further expensive function evaluation
            ###############################################################################
            select_indices = select_candidates(cand_values, cand_expl, opt, max_new_points, best_value,
                                               omega_percentage, tau_percentage, upsilon_percentage, sigma_t, sigma_u, rng)
            select_hyper = cand_hyper[select_indices, :]
            select_points = hypercube_to_arguments(select_hyper, arg_dict)
            select_sample = hypercube_to_sample(select_hyper, dim, arg_type, arg_array, arg_len)

            toc = time.perf_counter()               # time-reporting?
            if verbose_level >= 3:
                print(f'  timing: a subset of candidates selected in {toc - tic:.4f} seconds')
            tic = toc

            #########################################################################
            # compute the expensive function on the new candidate sample in parallel
            #########################################################################
            material = [(exp_func, point) for point in select_points]
            with mp.Pool(max(mp.cpu_count(), 1)) as pool:
                select_values = pool.map(compute_exp_func, material)

            evaluated += len(select_values)
            total_evaluated += len(select_values)

            toc = time.perf_counter()               # time-reporting?
            if verbose_level >= 3:
                print(f'  timing: a subset of candidates evaluated in {toc - tic:.4f} seconds')
            tic = toc

            ########################
            # iterative bookkeeping
            ########################
            computed_hyper = np.concatenate((computed_hyper, select_hyper))
            computed_sample = hypercube_to_sample(computed_hyper, dim, arg_type, arg_array, arg_len)
            # pd.concat can change dtypes from category to either int64 or object,
            # which will force xgboost to complain!
            # computed_sample = pd.concat([computed_sample, select_sample], ignore_index=True)
            computed_points = np.concatenate((computed_points, select_points))
            computed_values = np.concatenate((computed_values, select_values))

            index = opt_func(computed_values)
            best_point = computed_points[index]
            best_value = computed_values[index]
            best_sample = computed_sample.iloc[index]

            if verbose_level >= 2:              # report on the iterative sample?
                print(f'evaluated so far: {len(select_values)} selected, {evaluated} evaluated since last restart, {total_evaluated} in total')
                print(f'  best value={best_value} for point:  {best_point}')

            # perhaps we moved the total best value as well?
            if opt == 'max':
                if total_best_value is None or total_best_value < best_value:
                    total_best_value = best_value
                    total_best_point = best_point
            else:  # opt == 'min'
                if total_best_value is None or total_best_value > best_value:
                    total_best_value = best_value
                    total_best_point = best_point

            #########################################################
            # restart strategy: have we seen sufficient improvement?
            #########################################################
            if opt == 'max':
                if best_value > previous_best_value * (100+gamma)/100:
                    iter_last_improvement = iteration
                    previous_best_value = best_value
            else:
                if best_value < previous_best_value * (100-gamma)/100:
                    iter_last_improvement = iteration
                    previous_best_value = best_value

            if iteration - iter_last_improvement > r:
                # break out of the inner while total_evaluated < max_evaluations
                # to reach the outer while total_evaluated < max_evaluations
                # and restart the optimization process from scratch
                if verbose_level >= 2:
                    print(f'restart: no improvement since iteration {iter_last_improvement}, now iteration={iteration}')
                break

            #########################################################################
            # train XGBoost surrogate model for all values of the expensive function
            #########################################################################
            # do we use all data for training or do we split it into training and test set?
            if test_set_percentage < 1:
                xgb_model.fit(computed_sample, computed_values, verbose=False)
            else:
                rnd_indices = rng.permutation(len(computed_values))
                train_size = int(evaluated * (100 - test_set_percentage) / 100)

                # computed_sample is a pandas dataframe,
                # computed_values is a list

                train_sample = computed_sample.iloc[rnd_indices[0:train_size], :]
                train_values = [computed_values[i] for i in rnd_indices[0:train_size]]

                test_sample = computed_sample.iloc[rnd_indices[train_size:evaluated], :]
                test_values = [computed_values[i] for i in rnd_indices[train_size:evaluated]]

                xgb_model.fit(train_sample, train_values,
                              eval_set=[(test_sample, test_values)],
                              verbose=False)

            # continue training the previous XGBoost model?
            # select_sample = hypercube_to_sample(select_hyper, dim, arg_type, arg_array, arg_len)
            # xgb_model.fit(select_sample, select_values, verbose=False, xgb_model=xgb_model)

            toc = time.perf_counter()           # time-reporting?
            if verbose_level >= 3:
                print(f'  timing: new surrogate model trained in {toc - tic:.4f} seconds')
            tic = toc

            ###################################################
            # update the model for prediction of uncertainties
            ###################################################
            if uncertainty_metric == 'aposteriori':
                uncertainty_model.update(computed_sample, cand_values[select_indices], select_values)
            elif uncertainty_metric == 'voronoi':
                pass
            else:
                pass        # no need to update mipt uncertainty predictions...

            toc = time.perf_counter()           # time-reporting?
            if verbose_level >= 3:
                print(f'  timing: model for uncertainty predictions updated in {toc - tic:.4f} seconds')
            tic = toc

            ####################################################
            # csv file reporting for newly computed points only
            ####################################################
            if verbose_level >= 1:
                with open(filename, 'a') as f:
                    for i in range(len(select_indices)):
                        # iteration number
                        line = str(iteration)

                        # argument values for every computed point
                        for j in range(dim):
                            line += f', {select_sample.iat[i,j]}'

                        # computed function value for each point
                        line += f', {select_values[i]}'

                        # exploitation and exploration metrics
                        line += f', {cand_values[select_indices[i]]}'
                        line += f', {cand_expl[select_indices[i]]}'

                        # best value seen so far
                        line += f', {best_value}'

                        # argument values for best point so far
                        for j in range(dim):
                            line += f', {best_sample.iat[j]}'

                        f.write(line+'\n')

    # optimization complete!
    return total_best_value, total_best_point


if __name__ == '__main__':
    pass
