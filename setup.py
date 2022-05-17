from setuptools import setup, find_packages
from os.path import join
from config import PACKAGES


# Init DeepPhysX packages and dependencies to install
roots = ['Core']
available = {'ai': ['Torch'],
             'simu': ['Sofa']}
dependencies = {'Core': ['numpy', 'vedo', 'tensorboard', 'tensorboardX', 'pyDataverse'],
                'Sofa': [],
                'Torch': ['torch', 'psutil']}
packages = []
requires = []

# Include user config
user_ai_packages = []
user_simu_packages = []
for user_packages, key in zip([user_ai_packages, user_simu_packages], ['ai', 'simu']):
    for root in available[key]:
        if PACKAGES[root.lower()]:
            user_packages.append(root)

# Define the main packages to install
roots += user_ai_packages
roots += user_simu_packages

# Configure packages and subpackages list and dependencies list
prefix = 'DeepPhysX_'
packages = []
packages_dir = {}
requires = []
for root in roots:
    for sub_package in find_packages(where=root):
        if 'tests' not in sub_package:
            packages.append(sub_package)
            packages_dir[sub_package] = join(root, *sub_package.split('.'))
    requires += dependencies[root]

# Installation
setup(name='DeepPhysX',
      version='1.0',
      description='A Python framework interfacing AI with numerical simulation.',
      author='Mimesis',
      author_email='robin.enjalbert@inria.fr',
      url='https://github.com/mimesis-inria/DeepPhysX',
      packages=packages,
      package_dir=packages_dir,
      install_requires=requires,
      )
