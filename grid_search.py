from init import Dataset, CWD, ID_LIST
from sklearn import linear_model
from ransac import run_dataset, graph_roc_curves
from my_ransac_regressor import MyRANSACRegressor


def is_model_valid(model, X, y):
    try:
        coef = model.coef_
        try:
            if abs(coef[0]) > 3.0 or abs(coef[1]) > 0.06 or abs(coef[2]) > 0.04:
                return False
            else:
                return True

        except IndexError:
            return True

    except AttributeError:
        return True


estimators = [
    {
        'model': MyRANSACRegressor(base_estimator=linear_model.LinearRegression(n_jobs=-1),
                                   is_model_valid=is_model_valid),
        'id': 'MyRANSAC',
        'color': 'red',
        'grid_params': [
            {
                'max_trials': 1000,
                'min_samples': 5,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 10,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 25,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 50,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 75,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 100,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 125,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 150,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 175,
                'loss': 'absolute_loss'
            },
            {
                'max_trials': 1000,
                'min_samples': 200,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 'percentile',
                'percentile': 25,
                'max_trials': 1000,
                'min_samples': 100,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 'percentile',
                'percentile': 40,
                'max_trials': 1000,
                'min_samples': 100,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 'percentile',
                'percentile': 50,
                'max_trials': 1000,
                'min_samples': 100,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 'percentile',
                'percentile': 67,
                'max_trials': 1000,
                'min_samples': 100,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 'percentile',
                'percentile': 75,
                'max_trials': 1000,
                'min_samples': 100,
                'loss': 'absolute_loss'
            }            
        ]
    },
    {
        'model': linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(n_jobs=-1),
                                              is_model_valid=is_model_valid),
        'id': 'RANSAC & OLS',
        'color': 'orchid',
        'grid_params': [
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 5,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 10,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 20,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 50,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 75,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 100,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 125,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 150,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 175,
                'loss': 'absolute_loss'
            },
            {
                'residual_threshold': 0.4,
                'max_trials': 1000,
                'min_samples': 200,
                'loss': 'absolute_loss'
            }
        ]
    }
]


if __name__ == '__main__':
    for dataset_id in ID_LIST:
        dataset = Dataset(cwd=CWD, id=dataset_id)
        run_dataset(dataset, estimators, random_state=17)
        graph_roc_curves(dataset, estimators)
