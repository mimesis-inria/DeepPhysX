"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX_Core.Utils.data_downloader import DataDownloader

DOI = 'doi:10.5072/FK2/ZPFUBK'
session_name = 'liver_dpx'


class LiverDownloader(DataDownloader):

    def __init__(self):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models': [118, 120, 212],
                           'session': [281],
                           'network': [195],
                           'stats': [198],
                           'dataset_info': [276],
                           'dataset_valid': [279, 278],
                           'dataset_train': [277, 282, 280, 283]}


if __name__ == '__main__':

    LiverDownloader().get_session('valid_data')
