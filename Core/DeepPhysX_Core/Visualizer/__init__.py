from os.path import dirname
from os import listdir

package = dirname(__file__)
exceptions = ['__init__.py', '__pycache__']
modules = [module for module in listdir(package) if module.endswith('.py') and module not in exceptions]
__all__ = []
for module in sorted(modules):
    exec(f"from DeepPhysX_Core.Visualizer.{module[:-3]} import {module[:-3]}")
    __all__.append(module[:-3])
from DeepPhysX_Core.Visualizer import VedoObjectFactories
__all__ += VedoObjectFactories.__all__
