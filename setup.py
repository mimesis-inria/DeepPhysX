from setuptools import setup, find_packages
from os.path import join, pardir, exists
from json import load, dump


PROJECT = 'DeepPhysX'
PACKAGES = {'Torch': False,
            'Sofa': False}
DEPENDENCIES = {'Core': ['numpy', 'vedo', 'tensorboard', 'tensorboardX', 'pyDataverse'],
                'Sofa': [],
                'Torch': ['torch', 'psutil']}

# Load existing configuration file
if exists('config.json'):
    with open('config.json') as file:
        PACKAGES = load(file)
    # Check config validity
    correct_config = True
    for package_name, do_install in PACKAGES.items():
        if do_install and not exists(join(pardir, package_name)):
            PACKAGES[package_name] = False
            correct_config = False
    if not correct_config:
        with open('config.json', 'w') as file:
            dump(PACKAGES, file)

# Getting the packages to be installed
roots = ['Core']
for package_name, do_install in PACKAGES.items():
    if do_install:
        roots.append(package_name)

# Adding packages defined in configuration
packages = []
packages_dir = {}
requires = []
for package_name in roots:
    packages.append(f'{PROJECT}.{package_name}')
    packages_dir[f'{PROJECT}.{package_name}'] = join(pardir, package_name, 'src')
    requires += DEPENDENCIES[package_name]
    for sub_package in find_packages(where=join(pardir, package_name, 'src')):
        packages.append(f'{PROJECT}.{package_name}.{sub_package}')
        packages_dir[f'{PROJECT}.{package_name}.{sub_package}'] = join(pardir, package_name, 'src',
                                                                       *sub_package.split('.'))

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
