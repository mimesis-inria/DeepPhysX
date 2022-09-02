# SHARED: packages are installed within the shared namespace 'DeepPhysX'. Local install with pip.
# SINGLE: only Core package is installed with name 'DeepPhysX'. Distant install with pip.

from setuptools import setup, find_packages
from os.path import join, pardir, exists
from json import load, dump

PROJECT = 'DeepPhysX'
PACKAGE = 'Core'
PACKAGES = {'Torch': False,
            'Sofa': False}
DEPENDENCIES = {'Core': ['numpy', 'vedo', 'tensorboard', 'tensorboardX', 'pyDataverse'],
                'Sofa': [],
                'Torch': ['torch']}

# (SHARED) Loading existing configuration file
if exists('config.json'):
    with open('config.json') as file:
        PACKAGES = load(file)
    # Check config validity
    correct_config = True
    for package_name, do_install in PACKAGES.items():
        if do_install and not exists(join(pardir, package_name)):
            PACKAGES[package_name] = False
            correct_config = False
    # Write correction
    if not correct_config:
        with open('config.json', 'w') as file:
            dump(PACKAGES, file)

# (SINGLE / SHARED) Getting the packages to be installed
roots = [PACKAGE]
for package_name, do_install in PACKAGES.items():
    if do_install:
        roots.append(package_name)
packages = []
packages_dir = {}
requires = []

# (SINGLE) Specifying package list and corresponding directories
if len(roots) == 1:
    packages.append(f'{PROJECT}.{PACKAGE}')
    packages_dir[f'{PROJECT}.{PACKAGE}'] = 'src'
    requires += DEPENDENCIES[PACKAGE]
    for subpackage in find_packages(where='src'):
        packages.append(f'{PROJECT}.{PACKAGE}.{subpackage}')
        packages_dir[f'{PROJECT}.{PACKAGE}.{subpackage}'] = join('src', *subpackage.split('.'))

# (SHARED) Specifying package list and corresponding directories
else:
    for package_name in roots:
        packages.append(f'{PROJECT}.{package_name}')
        packages_dir[f'{PROJECT}.{package_name}'] = join(pardir, package_name, 'src')
        requires += DEPENDENCIES[package_name]
        for sub_package in find_packages(where=join(pardir, package_name, 'src')):
            packages.append(f'{PROJECT}.{package_name}.{sub_package}')
            packages_dir[f'{PROJECT}.{package_name}.{sub_package}'] = join(pardir, package_name, 'src',
                                                                           *sub_package.split('.'))

# (SINGLE / SHARED) Extract README.md content
with open('README.md') as f:
    long_description = f.read()

# (SINGLE / SHARED) Installation
setup(name='DeepPhysX',
      version='22.06',
      description='A Python framework interfacing AI with numerical simulation.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mimesis',
      author_email='robin.enjalbert@inria.fr',
      url='https://github.com/mimesis-inria/DeepPhysX',
      packages=packages,
      package_dir=packages_dir,
      install_requires=requires)
