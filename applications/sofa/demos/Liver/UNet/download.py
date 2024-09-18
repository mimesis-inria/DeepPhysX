"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX.Core.Utils.data_downloader import DataDownloader

DOI = 'doi:10.5072/FK2/PYMW0N'
session_name = 'liver_dpx'


class LiverDownloader(DataDownloader):

    def __init__(self):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models': [303, 304, 305],
                           'session': [290],
                           'network': [302],
                           'stats': [285],
                           'dataset_info': [333],
                           'dataset_valid': [336],
                           'dataset_train': [335, 334]}


if __name__ == '__main__':

    LiverDownloader().get_session('all')
