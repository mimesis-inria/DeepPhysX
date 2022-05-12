"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX_Core.Utils.data_downloader import DataDownloader

DOI = 'doi:10.5072/FK2/PYMW0N'
session_name = 'liver_dpx'


class LiverDownloader(DataDownloader):

    def __init__(self):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models': [303, 304, 307],
                           'session': [290],
                           'network': [302],
                           'stats': [285],
                           'dataset_info': [291],
                           'dataset_valid': [292, 296],
                           'dataset_train': [295, 286, 297, 299, 294, 287,
                                             289, 300, 288, 301, 298, 293]}


if __name__ == '__main__':

    # LiverDownloader().get_session('valid_data')
    LiverDownloader().show_content()
