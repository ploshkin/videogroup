import numpy as np
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn import linear_model


class MyRANSACRegressor(BaseEstimator):
    def __init__(self, base_estimator=None, min_samples=None,
                 residual_threshold='Ploshkin', percentile=60,
                 is_model_valid=None, max_trials=100, loss='absolute_loss'):

        self.base_estimator = base_estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.percentile = percentile
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.loss = loss


    def _score(self, distances):
        eps = np.amax(np.abs(distances))
        return (distances.shape[0] - np.sum(np.abs(distances)) / eps) / eps ** 2
        

    def fit(self, X, y):
        if self.base_estimator is not None:
            base_estimator = deepcopy(self.base_estimator)
        else:
            base_estimator = linear_model.LinearRegression()

        if self.min_samples is None:
            # assume linear model by default
            min_samples = X.shape[1] + 1
        elif self.min_samples >= 1:
            if self.min_samples % 1 != 0:
                raise ValueError("Absolute number of samples must be an "
                                 "integer value.")
            min_samples = self.min_samples
        else:
            raise ValueError("Value for `min_samples` must be scalar and "
                             "positive integer value.")
        if min_samples > X.shape[0]:
            min_samples = X.shape[0] / 2

        if 0 < self.percentile < 100:
            percentile = self.percentile
        else:
            raise ValueError("`percentile` must be float from (0, 100].")

        # adaptive residual threshold
        if self.residual_threshold == 'Ploshkin':
            residual_threshold = lambda \
                distances: np.maximum(0.05, np.abs(distances[distances < distances.std()]).mean())

        elif self.residual_threshold == 'percentile':
            residual_threshold = lambda \
                distances: np.percentile(np.abs(distances), percentile)

        elif callable(self.residual_threshold):
            residual_threshold = self.residual_threshold

        else:
            raise ValueError(
                "residual_threshold should be 'Ploshkin', 'percentile' or a callable."
                "Got %s. " % self.loss)


        if self.loss == "absolute_loss":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: y_true - y_pred
            else:
                loss_function = lambda \
                    y_true, y_pred: np.sum(np.abs(y_true - y_pred), axis=1)

        elif self.loss == "squared_loss":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: np.sign(y_true - y_pred) * (y_true - y_pred) ** 2
            else:
                loss_function = lambda \
                    y_true, y_pred: np.sum((y_true - y_pred) ** 2, axis=1)

        elif callable(self.loss):
            loss_function = self.loss

        else:
            raise ValueError(
                "loss should be 'absolute_loss', 'squared_loss' or a callable."
                "Got %s. " % self.loss)

        n_inliers_best = 0
        score_best = 0
        inlier_mask_best = None
        X_inlier_best = None
        y_inlier_best = None

        # number of data samples
        n_samples = X.shape[0]
        sample_idxs = np.arange(n_samples)

        n_samples, _ = X.shape

        for self.n_trials_ in range(1, self.max_trials + 1):

            # choose random sample set
            subset_idxs = np.random.choice(n_samples, min_samples, replace=False)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # fit model for current random sample set
            base_estimator.fit(X_subset, y_subset)

            # check if estimated model is valid
            if (self.is_model_valid is not None and not
                    self.is_model_valid(base_estimator, X_subset, y_subset)):
                continue

            # residuals of all data for current random sample model
            y_pred = base_estimator.predict(X)
            residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = np.abs(residuals_subset) < residual_threshold(residuals_subset)
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                continue
            if n_inliers_subset == 0:
                raise ValueError("No inliers found, possible cause is "
                    "setting residual_threshold too low.")

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = self._score(residuals_subset[inlier_idxs_subset])

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best
                    and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

        # if none of the iterations met the required criteria
        if inlier_mask_best is None:
            raise ValueError(
                "RANSAC could not find valid consensus set, because"
                " either the `residual_threshold` rejected all the samples or"
                " `is_model_valid` returned False for all"
                " `max_trials` randomly ""chosen sub-samples. Consider "
                "relaxing the ""constraints.")

        # estimate final model using all inliers
        base_estimator.fit(X_inlier_best, y_inlier_best)

        self.estimator_ = base_estimator
        self.inlier_mask_ = inlier_mask_best
        return self

    def predict(X):
        return self.estimator_.predict(X)
