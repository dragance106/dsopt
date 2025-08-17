"""
This file contains implementations of three classes that encapsulate
different ways of computing uncertainties for new candidate points.
"""
import numpy as np
import xgboost as xgb
from dsopt.mipt import inter_dist, proj_dist


class MIPTUncertainty:
    """
    Computes the uncertainty of new candidate points
    as a combination of intersite and projected distances from the existing points
    motivated by their use in Crombecq's MIPT sampling
    """
    def __init__(self, dim):
        self.dim = dim

    def predict(self, cand_hyper, computed_hyper):
        cand_inter = inter_dist(cand_hyper, computed_hyper)
        cand_proj = proj_dist(cand_hyper, computed_hyper)

        evaluated = len(computed_hyper)
        cand_expl = ((evaluated + 1) ** (1.0 / self.dim) - 1) / 2 * cand_inter + (evaluated + 1) / 2 * cand_proj

        return cand_expl

    def update(self):
        pass


class APosterioriUncertainty:
    """
    Computes the uncertainty of new candidate points
    by training an XGBoost model on historical relative errors
    between predicted and simulated values, and
    using this model to predict such relative errors for new candidate points.
    """

    # a small value to prevent division by zero when computing relative absolute errors
    EPSILON = 1e-10

    # the initial uncertainty measure represents
    # the relative absolute errors between the computed values and
    # the values predicted by two smaller xgboost models,
    # where the initial sample is divided into two halves,
    # with each model trained on one of these halves and used to predict the other half
    # the initial uncertainty model is then trained on these relative errors
    def __init__(self, init_sample, init_values, alpha):

        # the initial sample already consists of randomly selected points,
        # so just divide it into the first half and the second half
        k = len(init_sample)
        div1_sample = init_sample[0:k//2]
        div1_values = init_values[0:k//2]

        div2_sample = init_sample[k//2:k]
        div2_values = init_values[k//2:k]

        # train a separate model for each half
        uncertain_xgb1 = xgb.XGBRegressor(tree_method='approx',
                                          enable_categorical=True,
                                          max_cat_to_onehot=1,
                                          learning_rate=0.1,
                                          verbosity=0)
        uncertain_xgb1.fit(div1_sample, div1_values, verbose=False)

        uncertain_xgb2 = xgb.XGBRegressor(tree_method='approx',
                                          enable_categorical=True,
                                          max_cat_to_onehot=1,
                                          learning_rate=0.1,
                                          verbosity=0)
        uncertain_xgb2.fit(div2_sample, div2_values, verbose=False)

        # use the model trained on one half to predict the other half
        init_predictions = np.concatenate((uncertain_xgb2.predict(div1_sample),
                                           uncertain_xgb1.predict(div2_sample)))

        # compute the relative absolute errors between values and predictions
        self.rel_errors = np.abs(init_predictions - init_values) / (np.abs(init_values) + APosterioriUncertainty.EPSILON)

        # train the first uncertainty model on the initial relative absolute errors
        self.uncertain_xgb = xgb.XGBRegressor(tree_method='approx',
                                              enable_categorical=True,
                                              max_cat_to_onehot=1,
                                              learning_rate=0.1,
                                              verbosity=0)
        self.uncertain_xgb.fit(init_sample, self.rel_errors, verbose=False)

        # keep the copy of alpha for later use in update
        self.alpha = alpha

    # compute uncertainty predictions for the new candidate sample
    def predict(self, cand_sample):
        return self.uncertain_xgb.predict(cand_sample)

    def update(self, computed_sample, predicted_values, computed_values):
        # absolute relative errors of predictions for the newly computed values
        rel_errors_new = np.abs(predicted_values - computed_values) / (np.abs(computed_values) + APosterioriUncertainty.EPSILON)

        # train the uncertainty XGBoost model on
        # the combination of relative errors of predictions for the new values
        # and alpha times the relative errors of predictions for the previous values
        self.rel_errors = np.concatenate((self.alpha * self.rel_errors, rel_errors_new))
        self.uncertain_xgb.fit(computed_sample, self.rel_errors, verbose=False)


class VoronoiUncertainty:
    """
    Computes the uncertainty of new candidate points
    by training an XGBoost model on the estimates of volumes
    of Voronoi cell of the existing points, and
    then using this model to predict such estimates for new candidate points.
    """
    def __init__(self):
        # set up the xgboost model for learning Voronoi cell volume estimates
        self.uncertain_xgb = xgb.XGBRegressor(tree_method='approx',
                                              learning_rate=0.1,
                                              verbosity=0)

    def predict(self, cand_hyper, computed_hyper):
        # from each cand_hyper find the closest computed_hyper,
        # then update info on that computed_hyper
        voronoi_est = np.zeros(len(computed_hyper))
        for i, c in enumerate(cand_hyper):
            voronoi_est[np.argmin(np.sum((c - computed_hyper) ** 2, axis=1))] += 1

        # fit the model on those results for computed_hyper
        self.uncertain_xgb.fit(computed_hyper, voronoi_est, verbose=False)

        # then predict the values for cand_hyper
        return self.uncertain_xgb.predict(cand_hyper)

    def update(self):
        pass
