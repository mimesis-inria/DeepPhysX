from os.path import join
from subprocess import run

from setup_dev import check_repositories, define_config

PROJECT = 'DeepPhysX'
AI_PACKAGES = ['Torch']
SIMU_PACKAGES = ['Sofa']
PACKAGES = ['Core'] + AI_PACKAGES + SIMU_PACKAGES
GIT = {'Torch': 'https://github.com/mimesis-inria/DeepPhysX.Torch.git',
       'Sofa': 'https://github.com/mimesis-inria/DeepPhysX.Sofa.git'}


if __name__ == '__main__':

    # Check the project tree
    root = check_repositories()

    # Get the user configuration
    config = define_config(root)

    # Pip install packages
    for package in config:
        print(f"\nInstalling {f'{PROJECT}.{package}'} package...")
        run(['pip', 'install', '.'], cwd=join(root, package))
