from os.path import curdir, join, isdir
from os import mkdir
from json import load
import pandas as pd


class DirNames:
    def __init__(self):
        self.pts_dir = join(curdir, 'points')     
        self.gph_dir = join(curdir, 'graphs')
        self.res_dir = join(curdir, 'results')        
        self.inf_dir = join(curdir, 'info')


class IDList:
    def __init__(self, filename='dataset_names.json'):
        self.id_list = load(open(join(curdir, filename)))


class DatasetInit:
    def __init__(self, id):
        self.dir_names = DirNames()
        self.id = id

        self.pts = lambda video_name, scene_num=0: pd.read_csv(join(self.dir_names.pts_dir,
                                                                    self.id,
                                                                    '{}_pts.csv'.format(video_name)),
                                                               index_col='scene_num').loc[scene_num]

        self.roc_path = join(self.dir_names.gph_dir, 'ROC_{}.png'.format(self.id))
        self.prc_path = join(self.dir_names.gph_dir, 'precision_{}.png'.format(self.id))
        
        self.res_path = lambda estimator: join(self.dir_names.res_dir, self.id, '{}.csv'.format(estimator))


class Dataset(DatasetInit):
    def __init__(self, id):
        super().__init__(id)
        self.info = pd.read_csv(join(self.dir_names.inf_dir, '{}.csv'.format(self.id)), index_col='name')


def datasets_init(id_list):
    for id in id_list:
        dataset = DatasetInit(id)

        dataset.pts_dir = join(dataset.dir_names.pts_dir, dataset.id)
        if not isdir(dataset.pts_dir):
            mkdir(dataset.pts_dir)

        dataset.res_dir = join(dataset.dir_names.res_dir, dataset.id)
        if not isdir(dataset.res_dir):
            mkdir(dataset.res_dir)
        

if __name__ == '__main__':
    dir_names = DirNames()

    if not isdir(dir_names.pts_dir):
        mkdir(dir_names.pts_dir)

    if not isdir(dir_names.gph_dir):
        mkdir(dir_names.gph_dir)

    if not isdir(dir_names.res_dir):
        mkdir(dir_names.res_dir)

    if not isdir(dir_names.inf_dir):
        mkdir(dir_names.inf_dir)

    datasets_init(IDList(filename='dataset_names.json').id_list)
