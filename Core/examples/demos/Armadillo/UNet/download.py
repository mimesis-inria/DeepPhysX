"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX_Core.Utils.data_downloader import DataDownloader

DOI = 'doi:10.5072/FK2/MDQ46R'
session_name = 'armadillo_dpx'


class ArmadilloDownloader(DataDownloader):

    def __init__(self):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models': [142, 145, 306],
                           'session': [247],
                           'network': [230],
                           'stats': [224],
                           'dataset_info': [253],
                           'dataset_valid': [245, 257],
                           'dataset_train': [252, 255, 246, 256, 258, 251,
                                             250, 248, 254, 249, 259, 260]}


if __name__ == '__main__':

    ArmadilloDownloader().get_session('valid_data')
