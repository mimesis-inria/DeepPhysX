from os.path import join as osPathJoin
from os.path import isdir, abspath, normpath, dirname, basename
from os import listdir, pardir, makedirs
from inspect import getmodule, stack
from shutil import copytree


def create_dir(dir_path: str, dir_name: str) -> str:
    """
    Create a directory of the given name. If it already exist and specified, add a unique identifier at the end.

    :param str dir_path: Absolute directory to create
    :param str dir_name: Name of the directory to check for existence of a similar directories

    :return: Name of the created directory as string
    """
    if isdir(dir_path):
        print(f"Directory conflict: you are going to overwrite {dir_path}.")
        # Get the parent dir of training sessions
        parent = abspath(osPathJoin(dir_path, pardir))
        # Find all the duplicated folder
        deepest_repertory = dir_name.split('/')[-1] + '_'
        copies_list = [folder for folder in listdir(parent) if
                       isdir(osPathJoin(parent, folder)) and
                       folder.__contains__(deepest_repertory) and
                       folder.find(deepest_repertory) == 0 and
                       len(folder) in [len(deepest_repertory) + i for i in range(1, 4)]]
        # Get the indices of copies
        indices = [int(folder[len(deepest_repertory):]) for folder in copies_list]
        # The new copy is the max int + 1
        max_ind = max(indices) if len(indices) > 0 else 0
        new_name = basename(normpath(dir_path)) + f'_{max_ind + 1}/'
        dir_path = osPathJoin(parent, new_name)
        print(f"Create a new directory {dir_path} for this session.")
    makedirs(dir_path)
    return dir_path


def copy_dir(src_dir: str, dest_parent_dir: str, dest_dir: str) -> str:
    """
    Copy source directory to destination directory at the end of destination parent directory

    :param str src_dir: Source directory to copy
    :param str dest_parent_dir: Parent of the destination directory to copy
    :param str dest_dir: Destination directory to copy to

    :return: destination directory that source has been copied to
    """
    dest_dir = osPathJoin(dest_parent_dir, dest_dir)
    if isdir(dest_dir):
        print("Directory conflict: you are going to overwrite by copying in {}.".format(dest_dir))
        copies_list = [folder for folder in listdir(dest_parent_dir) if
                       isdir(osPathJoin(dest_parent_dir, folder)) and
                       folder.__contains__(dest_dir)]
        new_name = dest_dir + '({})/'.format(len(copies_list))
        dest_dir = osPathJoin(dest_parent_dir, new_name)
        print("Copying {} into the new directory {} for this session.".format(src_dir, dest_dir))
    else:
        new_name = dest_dir + '/'
        dest_dir = osPathJoin(dest_parent_dir, new_name)
    copytree(src_dir, dest_dir)
    return dest_dir


def get_first_caller() -> str:
    """
    Return the repertory in which the main script is
    """
    # Get the stack of called scripts
    scripts_list = stack()[-1]
    # Get the first one (the one launched by the user)
    module = getmodule(scripts_list[0])
    # Return the path of this script
    return dirname(abspath(module.__file__))

