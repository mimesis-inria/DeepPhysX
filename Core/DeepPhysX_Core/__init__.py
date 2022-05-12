from os.path import dirname
from os import listdir

package = dirname(__file__)
exceptions = ['__init__.py', '__pycache__']
modules = [module for module in listdir(package) if module not in exceptions]
__all__ = []
for module in sorted(modules):
    exec(f"from DeepPhysX_Core import {module}")
    __all__.append(module)
