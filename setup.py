from setuptools import setup, find_packages
from os.path import join

PROJECT = 'DeepPhysX'
PACKAGE = 'Core'

packages = [f'{PROJECT}.{PACKAGE}']
packages_dir = {f'{PROJECT}.{PACKAGE}': 'src'}

# Configure packages list and directories
for subpackage in find_packages(where='src'):
    packages.append(f'{PROJECT}.{PACKAGE}.{subpackage}')
    packages_dir[f'{PROJECT}.{PACKAGE}.{subpackage}'] = join('src', *subpackage.split('.'))

# Add examples as subpackages
packages.append(f'{PROJECT}.examples.{PACKAGE}')
packages_dir[f'{PROJECT}.examples.{PACKAGE}'] = 'examples'
for example_dir in find_packages(where='examples'):
    packages.append(f'{PROJECT}.examples.{PACKAGE}.{example_dir}')

# Extract README.md content
with open('README.md') as f:
    long_description = f.read()


def get_SSD():
    # If SSD was installed in dev mode, pip will re-install it
    try:
        import SSD
    except ModuleNotFoundError:
        return ['SimulationSimpleDatabase']
    return []


# Installation
setup(name='DeepPhysX',
      version='22.12',
      description='A Python framework interfacing AI with numerical simulation.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mimesis',
      author_email='robin.enjalbert@inria.fr',
      url='https://github.com/mimesis-inria/DeepPhysX',
      packages=packages,
      package_dir=packages_dir,
      install_requires=get_SSD() + ['numpy', 'tensorboard', 'tensorboardX', 'pyDataverse'])
