from os.path import curdir, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from template_dataset_estimator import DatasetInfo
from numpy.linalg import lstsq
from numpy.random import shuffle
from sklearn.decomposition import PCA


def ransac_score(x, y, k, b=0):
    norm = np.sqrt(1.0 + k ** 2)
    distances = (k * x + b - y) / norm
    abs_distances = np.abs(k * x + b - y) / norm
    eps = abs_distances[abs_distances < distances.std()].mean()
    if eps != 0:
        score = (1 / eps - abs_distances[abs_distances < eps] / (eps ** 2)).sum() / eps
    else:
        score = 10e6  # big value
    return eps, score


def make_hist(x, y, k, hist_path):
    norm = np.sqrt(1.0 + k ** 2)
    eps, score = ransac_score(x=x, y=y, k=k)

    edges = np.arange(-10, 10.2, 0.25)

    plt.clf()
    x_range = np.array([-10, 10])
    y_range = np.array([-10, 10])
    plt.hist2d(temp_y, disp_y, bins=edges, norm=Normalize(), cmap=plt.get_cmap('summer'), cmin=5)
    plt.plot(x_range * 3, k * y_range * 3, c='r')
    plt.plot(x_range * 3, k * y_range * 3 - eps * norm, c='b', linewidth=1.0)
    plt.plot(x_range * 3, k * y_range * 3 + eps * norm, c='b', linewidth=1.0)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.title('Temporal Shift')
    plt.xlabel('vertical motion vector ($temp_{y}$), px')
    plt.ylabel('compensated vertical disparity vector ($disp_{y}$), px')
    plt.colorbar()
    plt.text(x_range[0] + 1, y_range[-1] - 4, '$y\ =\ {:.4f}x$\nscore = {:.1f}\neps = {:.3f}'.format(k, score, eps),
             bbox={'facecolor': 'red', 'alpha': 0.1})
    plt.savefig(hist_path, fmt='png', dpi=600)


def ransac(points, n_iter=1000):
    best = pd.Series(np.zeros(6), index=['temporal_shift', 'angle', 'scale', 'const', 'score', 'eps'])
    indices = np.arange(len(points))
    for iter in range(n_iter):
        shuffle(indices)
        matrix = points.iloc[indices[:4]]
        matrix_Ak = np.ones((4,4))
        matrix_Ak[:,:-1] = matrix.drop('disp_y', axis=1).values
        matrix_bk = matrix.disp_y.values

        solution = pd.Series(lstsq(matrix_Ak, matrix_bk)[0], index=['temporal_shift', 'angle', 'scale', 'const'])
        if abs(solution.temporal_shift) <= 3.0:
            temp_y = points.temp_y
            disp_y = points.disp_y - points.vx * solution.angle - points.vy * solution.scale - solution.const
            eps, score = ransac_score(x=temp_y, y=disp_y, k=solution.temporal_shift)

            if score > best.score:
                best.temporal_shift = solution.temporal_shift
                best.angle = solution.angle
                best.scale = solution.scale
                best.const = solution.const
                best.score = score
                best.eps = eps

    return best


def ransac_pca(points, n_iter=1000):
    best = pd.Series(np.zeros(5), index=['temporal_shift', 'angle', 'scale', 'const', 'var_ratio'])
    best.var_ratio = 1

    pca = PCA(n_components=2)

    indices = np.arange(len(points))
    for iter in range(n_iter):
        shuffle(indices)
        # geometry parameters only
        matrix = points.iloc[indices[:3]]
        matrix_Ak = np.ones((3,3))
        matrix_Ak[:,:-1] = matrix.drop(['disp_y', 'temp_y'], axis=1).values
        matrix_bk = matrix.disp_y.values

        solution = pd.Series(lstsq(matrix_Ak, matrix_bk)[0], index=['angle', 'scale', 'const'])

        temp_y = points.temp_y
        disp_y = points.disp_y - points.vx * solution.angle - points.vy * solution.scale - solution.const

        X = np.array([temp_y, disp_y]).T
        pca.fit(X)
        PC = pca.transform(X)
        temporal_shift = pca.components_[0][1] / pca.components_[0][0]

        if pca.explained_variance_ratio_[1] < best.var_ratio and abs(temporal_shift) < 3.0:
            best.temporal_shift = temporal_shift
            best.angle = solution.angle
            best.scale = solution.scale
            best.const = solution.const
            best.var_ratio = pca.explained_variance_ratio_[1]

    print(best.temporal_shift)
    return best


def run_dataset_pca(dataset, metric='cur', n_iter=1000):
    ver = 'pca-based'
    np.random.seed(17)
    metric_name = dataset.metric_names[metric]
    results = pd.DataFrame(columns=['temporal_shift', 'angle', 'scale', 'const', 'var_ratio'])
    for scene_name in dataset.info.index:
        print(scene_name, end='\t')
        scene = dataset.info.loc[scene_name]
        points = pd.read_csv(join(dataset.pts_dir, metric_name, '{}_pts.csv'.format(scene.video_name)), index_col='scene_num').loc[0]
        results.loc[scene_name] = ransac_pca(points, n_iter)

    graph_precision(results, ground_truth=dataset.info['temporal shift'], version=ver)
    results.to_csv(join(curdir, '{}.csv'.format(ver)))


def graph_precision(results, ground_truth, version):
    plt.clf()

    precisions = np.arange(0, 0.52, step=0.01)
    diff = abs(results.temporal_shift - ground_truth)
    dataset_precisions = np.array([diff[diff <= eps].size / diff.size for eps in precisions]) * 100

    plt.plot(precisions, dataset_precisions, color='lime', linewidth=2.0, label=version, alpha=0.6)

    plt.xlim([0, 1.05 * precisions[-1]])
    plt.ylim([0, 105])

    plt.xlabel('max error, fr')
    plt.ylabel('proportion of correct results, %')
    plt.grid()
    plt.legend(loc='lower right')

    plt.tight_layout()

    plt.savefig(join(curdir, '{}_precision'.format(version)), fmt='png', dpi=600)


def cmp_precision(dataset):
    plt.clf()
    precisions = np.arange(0, 0.52, step=0.01)

    ground_truth = dataset.info['temporal shift']

    results = pd.read_csv(join(curdir, 'pre-alpha.csv'), index_col='name')

    diff = abs(results.temporal_shift - ground_truth)
    dataset_precisions = np.array([diff[diff <= eps].size / diff.size for eps in precisions]) * 100

    plt.plot(precisions, dataset_precisions, color='k', linewidth=2.0, label='pre-alpha', alpha=0.6)

    for metric in dataset.metric_names.keys():
        res = pd.DataFrame(columns=['name', 'const', 'temporal_shift', 'angle', 'scale']).set_index('name')
        for scene_name in dataset.info.index:
            scene = dataset.info.loc[scene_name]
            video_name = scene.video_name
            res.loc[scene_name] = pd.read_csv(join(dataset.sln_dir, dataset.metric_names[metric], '{}_sln.csv'.format(video_name)), index_col='scene_num').loc[0]

        diff = abs(res.temporal_shift - ground_truth)
        dataset_precisions = np.array([diff[diff <= eps].size / diff.size for eps in precisions]) * 100

        plt.plot(precisions, dataset_precisions, color=dataset.colors[metric], linewidth=2.0, label=dataset.metric_names[metric], alpha=0.6)

    plt.xlim([0, 1.05 * precisions[-1]])
    plt.ylim([0, 105])

    plt.xlabel('max error, fr')
    plt.ylabel('proportion of correct results, %')
    plt.grid()
    plt.legend(loc='lower right')

    plt.tight_layout()

    plt.savefig(join(dataset.graphs_dir, 'cmp_precision'), fmt='png', dpi=600)


def run_dataset(dataset, metric='cur', n_iter=1000):
    ver = 'pre-alpha'
    np.random.seed(17)
    metric_name = dataset.metric_names[metric]
    results = pd.DataFrame(columns=['name', 'temporal_shift', 'angle', 'scale', 'const', 'score', 'eps']).set_index('name')
    for scene_name in dataset.info.index:
        print(scene_name)
        scene = dataset.info.loc[scene_name]
        points = pd.read_csv(join(dataset.pts_dir, metric_name, '{}_pts.csv'.format(scene.video_name)), index_col='scene_num').loc[0]
        results.loc[scene_name] = ransac(points, n_iter)

    # graph_precision(results, ground_truth=dataset.info['temporal shift'], version=ver)
    results.to_csv(join(curdir, '{}.csv'.format(ver)))


dataset = DatasetInfo()
# run_dataset(dataset, n_iter=1000)
# cmp_precision(dataset)
