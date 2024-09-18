"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX.Core.Utils.data_downloader import DataDownloader

DOI = 'doi:10.5072/FK2/PJUE43'
session_name = 'beam_dpx'


class BeamDownloader(DataDownloader):

    def __init__(self):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models': [308],
                           'session': [312],
                           'networks': [315],
                           'stats': [314],
                           'dataset_info': [329],
                           'dataset_valid': [328],
                           'dataset_train': [327]}


if __name__ == '__main__':

    BeamDownloader().get_session('all')
