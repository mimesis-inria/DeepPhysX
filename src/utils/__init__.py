# # from os import listdir
# # from os.path import isdir, isfile, join, sep
# # from pathlib import Path
# #
# # current_absolute_path = Path(__file__).parent.absolute()
# # current_relative_directory = str(Path(__file__).parent).split(sep)[-1]
# #
# # for object in listdir(current_absolute_path):
# #     # Ignore current and any __pycache__ directories
# #     if isdir(object) and not "__pycache__" in object:
# #         exec(f"from {current_relative_directory} import {object}")
# #     # Import python file that is not this one
# #     elif isfile(object) and ".py" in object and not "__init__.py" == object:
# #         exec(f"import {object}")# from os import listdir
# # # from os.path import isdir, isfile, join, sep
# # # from pathlib import Path
# # #
# # # current_absolute_path = Path(__file__).parent.absolute()
# # # current_relative_directory = str(Path(__file__).parent).split(sep)[-1]
# # #
# # # for object in listdir(current_absolute_path):
# # #     # Ignore current and any __pycache__ directories
# # #     if isdir(object) and not "__pycache__" in object:
# # #         exec(f"from {current_relative_directory} import {object}")
# # #     # Import python file that is not this one
# # #     elif isfile(object) and ".py" in object and not "__init__.py" == object:
# # #         exec(f"import {object}")
#
# from DeepPhysX_Core.utils.mathUtils import *
# from DeepPhysX_Core.utils.pathUtils import *
# from DeepPhysX_Core.utils.tensor_transform_utils import *
#
# __all__ = ["nextPowerOf2",
#            "fibonacci3DSphereSampling",
#            "sigmoid",
#            "min_max_feature_scaling",
#            "ndim_interpolation",
#            "createDir",
#            "copyDir",
#            "getFirstCaller",
#            "flatten"]