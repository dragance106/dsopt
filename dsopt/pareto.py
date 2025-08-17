"""
Selection of candidate points for expensive function evaluation
as a mix of candidates with low uncertainty and best predictions,
Pareto solutions among candidates with medium uncertainty,
and a handful of candidates with highest uncertainty.
"""
import numpy as np
import pandas as pd


def select_candidates(cand_values, cand_expl, opt, max_new_points,
                      best_simulated_value, omega_percentage = 5,
                      tau_percentage=10, upsilon_percentage=25,
                      sigma_t = 37.5, sigma_u = 12.5,
                      rng=np.random.default_rng()):
    """
    Selects a subset of candidates for further expensive function evaluation
    by combining candidates with best predicted values among the candidates with low uncertainties,
    Pareto solutions for uncertainties and predicted values among the candidates with medium uncertainties, and
    the candidates with the highest uncertainties among the candidates with high uncertainties.

    Low uncertainty is represented by the candidates with tau_percentage lowest uncertainties,
    while high uncertainty is represented by the candidates with upsilon_percentage highest uncertainties.
    The remaining candidates are classified as medium uncertainty.

    Low uncertainty candidates have to have their predicted value
    not lagging by more than omega% from the best simulated value
    in order to be selected for further expensive function evaluation.

    Pareto solutions among medium uncertainty candidates also have to have their predicted value
    not lagging by more than omega% from the best simulated value
    in order to be selected for further expensive function evaluation.
    In case there are more Pareto solutions than available spots,
    Pareto solutions are selected uniformly at random.

    :param cand_values:         predicted values of the candidate points
    :param cand_expl:           uncertainty metric/exploration measure of the candidate points
    :param opt:                 'min' or 'max', determining the objective for predicted values
    :param max_new_points:      maximum number of points to be selected for further expensive function evaluation
    :param best_simulated_value:
                                the best simulated value obtained so far
    :param omega_percentage:    the allowed percentage of lagging behind best_simulated_value
    :param tau_percentage:      this percentage of lowest uncertainties determine low uncertainty candidates
    :param upsilon_percentage:  this percentage of highest uncertainties determine high uncertainty candidates
    :param sigma_t:             percentage of selected candidates with low uncertainty
    :param sigma_u:             percentage of selected candidates with high uncertainty
    :param rng:                 random number generator, if specifically seeded generator is needed
    :return:                    the indices of selected candidates for further expensive function evaluation

    """

    # cand_expl are always to be maximized
    # cand_values are to be minimized if opt=='min' and maximized if opt=='max',
    # so unify the objectives
    if opt == 'min':
        cand_opt_values = -cand_values
        best_prediction_cutoff = -best_simulated_value * (100-omega_percentage)/100
    else:
        cand_opt_values = cand_values
        best_prediction_cutoff = best_simulated_value * (100-omega_percentage)/100
    # now both cand_expl and cand_opt_values are to be maximized

    # join opt_values, exploration and indices together in a new dataframe
    df = pd.DataFrame({'values': cand_opt_values,                                   # column 0
                       'expl': cand_expl,                                           # column 1
                       'orig_indices': np.arange(len(cand_opt_values), dtype=int)}) # column 2
    df['pareto'] = True             # 'pareto' is column 3
                                    # 'orig_indices' column is not really necessary,
                                    # as it's always the same as the index, but who cares...
    # sort the candidates by increasing expl
    df.sort_values(by=['expl', 'values'], ascending=[True, False], inplace=True)

    # low uncertainty = tau% lowest uncertainties
    # select sigma_t% points with best predictions
    low_boundary = round(tau_percentage * len(cand_opt_values) / 100)
    select_low_best = min(low_boundary, round(0.5 + sigma_t * max_new_points/100))
    # retain only those whose predicted value is not worse than best_prediction_cutoff?
    # df_low = df_low[df_low['values']>=best_prediction_cutoff]
    low_indices = df[:low_boundary].nlargest(select_low_best, 'values')['orig_indices'].to_numpy(dtype=int)
    df.iloc[:low_boundary, 3] = False

    # high uncertainty = upsilon% highest uncertainties
    # select sigma_u% of those with highest uncertainty
    high_boundary = max(round(upsilon_percentage * len(cand_opt_values) / 100), 1)
    select_high_best = max(1, min(high_boundary, round(sigma_u * max_new_points/100)))
    high_indices = df[-high_boundary:].nlargest(select_high_best, 'expl')['orig_indices'].to_numpy(dtype=int)
    df.iloc[-high_boundary:, 3] = False

    # medium uncertainty is everything else
    # skips first low_boundary and last high_boundary

    # determine Pareto solutions, which are those candidates
    # for which no other candidate has both smaller uncertainty and larger value
    # candidates are already sorted by expl in ascending and then by values in descending order
    # so candidate A that appears earlier than candidate B already has smaller expl
    # hence for each candidate B we have to check
    # whether the maximum value of earlier candidates is larger than its value
    # (in which case B is dominated and not Pareto solution)

    prev_max_value = df.iat[low_boundary, 0]
    for i in range(1, len(cand_opt_values)-high_boundary-low_boundary):
        if prev_max_value > df.iat[low_boundary+i, 0]:
            df.iat[low_boundary+i, 3] = False
        prev_max_value = max(prev_max_value, df.iat[low_boundary+i, 0])

    # extract the original indices of Pareto solutions
    pareto_indices = df[df['pareto'] == True]['orig_indices'].to_numpy(dtype=int)

    # if there are not enough Pareto solutions available, select them all
    remaining_points = max(0, max_new_points - len(low_indices) - len(high_indices))
    if len(pareto_indices) <= remaining_points:
        medium_indices = pareto_indices
    else:
        # otherwise select Pareto solutions uniformly at random, or equivalently,
        # randomly permute the indices and then take those at the beginning
        medium_indices = pareto_indices[rng.permutation(len(pareto_indices))[:remaining_points]]

    selected_indices = np.concatenate([low_indices, medium_indices, high_indices], axis=None)
    return selected_indices


def select_candidates_old(cand_values, cand_expl, opt, max_new_points, first_uniform_percentage=75):
    """
    Selects a subset of candidates for further expensive function evaluation
    by combining first_uniform_percentage of candidates with best predicted value
    and (100-first_uniform_percentage) of Pareto solutions for the exploitation/exploration metrics.

    :param cand_values:     predicted values for the new candidate points
    :param cand_expl:       exploration measure for the new candidate points
    :param opt:             'min' or 'max', determining the objective for predicted values
    :param max_new_points:  maximum number of Pareto solutions to be selected for expensive function computation
    :param first_uniform_percentage:
                            determines how the Pareto solutions are selected:
                            'uniform'-ly or the 'first' ones with optimum cand_values
    :return:                the indices of selected solutions for further evaluation
    """

    # cand_expl are always to be maximized
    # cand_values are to be minimized if opt=='min' and maximized if opt=='max',
    # so unify the objectives
    if opt == 'min':
        cand_opt_values = -cand_values
    else:
        cand_opt_values = cand_values
    # now both cand_expl and cand_opt_values are to be maximized

    # join opt_values, exploration and indices together in a new dataframe
    ser_values = pd.Series(cand_opt_values)
    ser_expl = pd.Series(cand_expl)
    df_values_expl = pd.DataFrame({'values': ser_values, 'expl': ser_expl})

    # sort the candidates by values first, then by expl
    df_values_expl.sort_values(by=['values', 'expl'], inplace=True, ascending=False)

    # first select a handful of candidates with best values
    # (index shows the indices from the original array)
    first_points = (max_new_points * first_uniform_percentage) // 100
    first_indices = df_values_expl.iloc[0:first_points].index.to_numpy()
    cutoff_value = df_values_expl.iat[first_points-1, 0]

    # for the rest select uniformly among the Pareto solutions with smaller values
    # Pareto solutions are those for which there is no other solution with larger both value and expl
    # solutions are already sorted first by values, then by expl,
    # so value is already larger if another solution appears earlier in the sorted list
    # hence for each solution we have to check
    # if the maximum expl of the previous solutions is larger than the current expl
    df_values_expl['pareto'] = True                                 # 0 is column 'values'
    prev_max_expl = df_values_expl.iloc[0, 1]                       # 1 is column 'expl'
    for i in range(1, len(df_values_expl.index)):                   # 2 is column 'pareto'
        if prev_max_expl > df_values_expl.iloc[i, 1]:
            df_values_expl.iloc[i, 2] = False
        prev_max_expl = max(prev_max_expl, df_values_expl.iloc[i, 1])

    # extract the indices of Pareto solutions with values smaller than the cutoff value
    pareto_indices = df_values_expl[(df_values_expl['pareto'] == True) &
                                    (df_values_expl['values'] < cutoff_value)].index.to_numpy()

    remaining_points = max_new_points - first_points
    if len(pareto_indices) <= remaining_points:
        # if there are not too many Pareto solutions available,
        # they all get added to the first points
        select_indices = np.concatenate([first_indices, pareto_indices])
    else:
        # otherwise, select uniformly among Pareto solutions
        keep_indices = np.floor(np.linspace(0, len(pareto_indices)-1, num=remaining_points)).astype(int, casting='unsafe')
        select_indices = np.concatenate([first_indices, pareto_indices[keep_indices]])

    return select_indices
