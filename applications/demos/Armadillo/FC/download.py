"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX.Core.Utils.data_downloader import DataDownloader

DOI = 'doi:10.5072/FK2/B1NUY0'
session_name = 'armadillo_dpx'


class ArmadilloDownloader(DataDownloader):

    def __init__(self):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models':  [111, 112, 210],
                           'session': [240],
                           'network': [193],
                           'stats':   [192],
                           'dataset_info':  [317],
                           'dataset_valid': [319],
                           'dataset_train': [318]}


if __name__ == '__main__':

    ArmadilloDownloader().get_session('all')
