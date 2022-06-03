from os.path import dirname
from os import listdir

package = dirname(__file__)
exceptions = ['__init__.py', '__pycache__']
modules = [module for module in listdir(package) if module.endswith('.py') and module not in exceptions]
__all__ = []
for module in sorted(modules):
    exec(f"from DeepPhysX.Core.Visualizer.VedoObjectFactories.{module[:-3]} import {module[:-3]}")
    __all__.append(module[:-3])
