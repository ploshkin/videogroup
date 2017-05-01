import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from json import dump


def make_hist(x, y, k, hist_path):
    edges = np.arange(-10, 10.25, 0.25)

    plt.clf()
    x_range = np.array([-10, 10])
    y_range = np.array([-10, 10])
    plt.hist2d(temp_y, disp_y, bins=edges, norm=Normalize(), cmap=plt.get_cmap('summer'), cmin=5)
    plt.plot(x_range * 3, k * y_range * 3, c='r')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.title('Temporal Shift')
    plt.xlabel('vertical motion vector ($temp_{y}$), px')
    plt.ylabel('compensated vertical disparity vector ($disp_{y}$), px')
    plt.colorbar()
    plt.text(x_range[0] + 1, y_range[-1] - 4, '$y\ =\ {:.4f}x$'.format(k),
             bbox={'facecolor': 'red', 'alpha': 0.1})
    plt.savefig(hist_path, fmt='png', dpi=600)


def graph_roc_curves(dataset, estimators):
    plt.clf()
    precisions = np.arange(0, 0.52, step=0.01)

    ground_truth = dataset.info['temporal shift']

    for estimator_info in estimators:
        results = pd.read_csv(dataset.res_path(estimator_info['id']), index_col='name')

        diff = abs(results.temporal_shift - ground_truth)
        dataset_precisions = np.array([diff[diff <= eps].size / diff.size for eps in precisions]) * 100

        plt.plot(precisions, dataset_precisions,
                 color=estimator_info['color'], label=estimator_info['id'], linewidth=2.0, alpha=0.6)

    plt.xlim([0, 1.05 * precisions[-1]])
    plt.ylim([0, 105])

    plt.xlabel('max error, fr')
    plt.ylabel('proportion of correct results, %')
    plt.grid()
    plt.legend(loc='lower right')

    plt.tight_layout()

    plt.savefig(dataset.roc_path, fmt='png', dpi=600)


def run_dataset(dataset, estimators, random_state=None):
    cols = ['name', 'temporal_shift', 'angle', 'scale', 'const']
    for estimator_info in estimators:
        best_estimator = grid_search(dataset, estimator_info, random_state)
        best_estimator['results'].to_csv(dataset.res_path(estimator_info['id']))
        dump({'params': best_estimator['params'], 'AUC': best_estimator['AUC']},
             open(dataset.prm_path(estimator_info['id']), 'w'), indent=4)


def AUC_score(X, y):
    h = 0.01
    precisions = np.arange(0, 1.0 + h, step=h)
    diff = abs(X - y)
    dataset_precisions = np.array([diff[diff <= eps].size / diff.size for eps in precisions])

    AUC = (dataset_precisions[1:-2].sum() + (dataset_precisions[0] + dataset_precisions[-1]) / 2) * h
    return AUC


def grid_search(dataset, estimator_info, random_state=None):
    if random_state:
        np.random.seed(random_state)

    cols = ['name', 'temporal_shift', 'angle', 'scale']

    best_estimator = {
        'params' : None,
        'results': None,
        'AUC'    : 0
    }

    estimator = estimator_info['model']

    for params in estimator_info['grid_params']:
        estimator.set_params(**params)

        results = pd.DataFrame(columns=cols).set_index('name')
        for scene_name in dataset.info.index:
            print(scene_name)
            scene = dataset.info.loc[scene_name]
            points = dataset.pts(video_name=scene.video_name)

            X = points.drop('disp_y', axis=1).values
            y = points.disp_y.values

            try:
                estimator.fit(X, y)

                try:
                    coef = estimator.estimator_.coef_
                except AttributeError:
                    coef = estimator.coef_

                results.loc[scene_name] = pd.Series(coef, index=['temporal_shift', 'angle', 'scale'])

            except:
                pass

        AUC = AUC_score(results.temporal_shift, dataset.info['temporal shift'])
        print(AUC)
        if AUC > best_estimator['AUC']:
            best_estimator['params']  = params
            best_estimator['results'] = results
            best_estimator['AUC']     = AUC

    return best_estimator
