import os
from pyDataverse.api import NativeApi, DataAccessApi


class DataDownloader:

    def __init__(self, DOI, session_name, session_dir='sessions'):

        # Connect to Dataverse API
        base_url = 'https://data-qualif.loria.fr'
        self.api = NativeApi(base_url)
        self.data_api = DataAccessApi(base_url)

        # Get files data for desired Dataset
        self.filenames = {}
        for file in self.api.get_dataset(DOI).json()['data']['latestVersion']['files']:
            self.filenames[file['dataFile']['id']] = file['dataFile']['filename']

        # Sessions repositories and content
        self.root = os.getcwd()
        self.session_dir = session_dir
        self.session_name = session_name
        self.categories = {'models':  [],
                           'session': [],
                           'network': [],
                           'stats':   [],
                           'dataset_info':  [],
                           'dataset_valid': [],
                           'dataset_train': []}
        self.sessions = {'run':        ['models'],
                         'train':      ['models'],
                         'train_data': ['models', 'session', 'dataset_info', 'dataset_train'],
                         'valid':      ['models', 'session', 'dataset_info', 'network', 'stats'],
                         'valid_data': ['models', 'session', 'dataset_info', 'network', 'stats', 'dataset_valid'],
                         'predict':    ['models', 'session', 'dataset_info', 'network', 'stats'],
                         'all': list(self.categories.keys())}
        self.tree = {'models': 'Environment/models',
                     'session': f'{session_dir}/{session_name}',
                     'network': f'{session_dir}/{session_name}/network',
                     'stats':   f'{session_dir}/{session_name}/stats',
                     'dataset': f'{session_dir}/{session_name}/dataset'}
        self.nb_files = 0

    def show_content(self):

        # Print the content of the desired Dataset
        for file_id in sorted(list(self.filenames.keys())):
            print(f"\t{file_id}: {self.filenames[file_id]}")

    def get_session(self, session_name):

        # Check session tree
        if session_name not in self.sessions.keys():
            raise ValueError(f'[DataDownloader] Session name must be in {self.sessions.keys()}')
        self.check_tree(session_name)

        # Download missing files
        if self.nb_files > 0:
            print("Connecting to DeepPhysX online storage to download missing data...")
            self.download_files(session_name)
            self.nb_files = 0

    def check_tree(self, session_name):

        # Check for each data category
        for category in self.sessions[session_name]:

            # Only 'models' is created in an existing directory
            if category != 'models':

                # Check session repository
                if not os.path.exists(os.path.join(self.root, self.session_dir)):
                    os.mkdir(os.path.join(self.root, self.session_dir))
                if not os.path.exists(os.path.join(self.root, self.tree['session'])):
                    os.mkdir(os.path.join(self.root, self.tree['session']))

            # Check the last level of subdirectories
            last_level = 'dataset' if 'dataset' in category else category
            if not os.path.exists(os.path.join(self.root, self.tree[last_level])):
                os.mkdir(os.path.join(self.root, self.tree[last_level]))

            # Get the number of file to download
            for file in self.categories[category]:
                if not os.path.exists(os.path.join(self.root, self.tree[last_level], self.filenames[file])):
                    self.nb_files += 1

    def download_files(self, session_name):

        # Download each missing file of the required categories
        nb_file = 0
        for category in self.sessions[session_name]:
            for file in self.categories[category]:
                last_level = 'dataset' if 'dataset' in category else category
                if not os.path.exists(os.path.join(self.root, self.tree[last_level], self.filenames[file])):
                    nb_file += 1
                    print(f"\tDownloading file {nb_file}/{self.nb_files} in "
                          f"{os.path.join(self.tree[last_level], self.filenames[file])}")
                    data = self.data_api.get_datafile(file)
                    with open(os.path.join(self.root, self.tree[last_level], self.filenames[file]), 'wb') as f:
                        f.write(data.content)
