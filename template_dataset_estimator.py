###############
# VERSION 1.2 #
###############

from os.path import curdir, join, relpath
from shutil import move, copyfile
import pandas as pd
from subprocess import call
from json import load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def source_for_avs(video):
    fmt = video.split('.')[-1]
    if fmt == 'avi' or fmt == 'avs':
        res = 'Avisource("{}")'.format(video)
    elif fmt == 'mkv':
        res = 'ffvideosource("{}")'.format(video)
    else:
        print('Not supported format: {}'.format(fmt))
        res = None
    return res


def create_avs(src_left, src_right, dst_dir):
    rel_left = relpath(src_left, dst_dir)
    rel_right = relpath(src_right, dst_dir)

    left = '{}.Lanczos4Resize(960, 544).ConvertToRGB24()\n'.format(source_for_avs(rel_left))
    right = '{}.Lanczos4Resize(960, 544).ConvertToRGB24()\n'.format(source_for_avs(rel_right))

    with open(join(dst_dir, 'left.avs'), 'w') as left_avs:
        left_avs.write(left)

    with open(join(dst_dir, 'right.avs'), 'w') as right_avs:
        right_avs.write(right)


class DatasetInfo:
    def __init__(self, dataset_info_file='dataset_info.json'):
        # Basic dataset info contains:
        #    * file with dataset info table
        #    * index column name of dataset info table
        #    * current and new metric titles
        basic_info = load(open(join(curdir, dataset_info_file)))

        # Directories
        self.src_dir = join(curdir, 'source_video')
        self.avs_dir = join(curdir, 'avs')
        self.sc_dir = join(curdir, 'sc')
        self.estimated_dir = join(curdir, 'estimated_shift')
        self.analysis_dir = join(curdir, 'analysis')
        self.pts_dir = join(self.analysis_dir, 'points')
        self.sln_dir = join(self.analysis_dir, 'solutions')
        self.hist_dir = join(self.analysis_dir, 'histograms')
        self.graphs_dir = join(self.analysis_dir, 'graphs')
        self.metrics_dir = join(curdir, 'metrics')

        # Dataset info table
        self.info = pd.read_csv(join(curdir, basic_info['dataset_details'])).set_index(basic_info['index_column'])

        # Metric titles
        self.metric_names = {'old': 'temporal_asynchrony_metric',
                             'cur': basic_info['cur_metric_name'],
                             'new': basic_info['new_metric_name']}

        self.colors = {'old': 'red', 'cur': 'blue', 'new': 'lime'}


class DatasetEstimator:
    def __init__(self):
        self.dataset = DatasetInfo(dataset_info_file='dataset_info.json')

    def evaluate_scene(self, video_name, metric_name, fmt='avi'):
        metric_dir = join(self.dataset.metrics_dir, metric_name)
        args = [join(metric_dir, 'samplehost.exe'), '-w', '960', '-h', '544',
                '-i', 'crosstalk_test\\left.avs', 'crosstalk_test\\right.avs',
                '-b', '20', '-s', '1', '-o', 'out_async', '-m', 'asynchrony']

        src_left = join(self.dataset.src_dir, '{}_left.{}'.format(video_name, fmt))
        src_right = join(self.dataset.src_dir, '{}_right.{}'.format(video_name, fmt))
        dst = join(metric_dir, 'crosstalk_test')

        # file with scene changes
        copyfile(join(self.dataset.sc_dir, '{}_sc.txt'.format(video_name)),
                 join(metric_dir, 'sc.txt'))

        create_avs(src_left, src_right, dst)
        call(args, cwd=metric_dir)

        # file with points for RANSAC
        move(join(metric_dir, 'points.csv'), join(self.dataset.pts_dir, metric_name, '{}_pts.csv'.format(video_name)))

        # file with solutions
        move(join(metric_dir, 'solutions.csv'), join(self.dataset.sln_dir, metric_name, '{}_sln.csv'.format(video_name)))

        # file with resulting asynchrony for each frame
        move(join(metric_dir, 'out_async', 'asynchrony.txt'),
             join(self.dataset.estimated_dir, metric_name, '{}_async.txt'.format(video_name)))

    def run_metric(self, metric):
        metric_name = self.dataset.metric_names[metric]
        for video_name, indices in self.dataset.info.groupby('video_name').groups.items():
            print(video_name)
            fmt = self.dataset.info.loc[indices[0]].fmt
            self.evaluate_scene(video_name, metric_name, fmt)

    def make_hist(self, scene_name, video_name, metric, scene_num):
        metric_name = self.dataset.metric_names[metric]
        solution = pd.read_csv(join(self.dataset.sln_dir, metric_name, '{}_sln.csv'.format(video_name)),
                               index_col='scene_num').loc[scene_num]
        k = solution.temporal_shift

        points = pd.read_csv(join(self.dataset.pts_dir, metric_name, '{}_pts.csv'.format(video_name)),
                             index_col='scene_num')
        points = points[points.index == scene_num]

        temp_y = points.temp_y
        if metric == 'old':
            disp_y = points.disp_y - solution.const
        else:
            disp_y = points.disp_y - points.vx * solution.angle - points.vy * solution.scale - solution.const

        edges = np.arange(-10, 10.2, 0.25)

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
        plt.text(x_range[0] + 1, y_range[1] - 2, '$y\ =\ {:.4f}x$'.format(k),
                 bbox={'facecolor': 'red', 'alpha': 0.1})
        plt.savefig(join(self.dataset.hist_dir, '{}_{}'.format(scene_name, metric)), fmt='png', dpi=600)

    def graph_hists_for_metric(self, metric):
        dataset_size = len(self.dataset.info.index)
        for num, scene_name in enumerate(self.dataset.info.index):
            print('{} / {}'.format(num + 1, dataset_size))
            video_name = self.dataset.info.loc[scene_name]['video_name']
            try:
                scene_num = self.dataset.info.loc[scene_name]['scene_num_in_fragment']
            except KeyError:
                scene_num = 0

            self.make_hist(scene_name, video_name, metric, scene_num)
