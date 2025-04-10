from typing import Optional
from os.path import join, isdir, abspath, dirname, isabs
from os import listdir, makedirs
from inspect import getmodule, stack
from shutil import copytree


def get_first_caller() -> str:
    """
    Return the repertory in which the main script is stored.
    """

    # Get the stack of calls
    scripts_list = stack()[-1]
    # Get the first one (the one launched by the user)
    module = getmodule(scripts_list[0])
    # Return the path of this script
    return dirname(abspath(module.__file__))


def get_session_dir(session_dir: str,
                    new_session: bool) -> str:
    """
    Get the DPX session root directory.

    :param session_dir: Path to the current session.
    :param new_session: If True, this session is a new one.
    """

    # 1. Check if the given path is absolute
    if isabs(session_dir):
        # For an existing session, this sessions dir must exist
        if not new_session and not isdir(session_dir):
            raise ValueError(f"Impossible to locate the repository containing your sessions. \n"
                             f"   - {session_dir} not found ('session_dir' variable as absolute path).")
        return session_dir

    # 2. The path is relative, make it absolute for a new session
    if new_session:
        return join(get_first_caller(), session_dir)

    # 3. The path is relative, find the absolute path for an existing session
    else:
        # 3.1. Relative path from the working directory
        if isdir(rel_wd := abspath(session_dir)):
            return rel_wd
        # 3.2. Relative path from the script directory
        elif isdir(rel_sd := join(get_first_caller(), session_dir)):
            return rel_sd
        # 3.3. Impossible to find the existing sessions repository
        else:
            raise ValueError(f"Impossible to locate the repository containing your sessions. \n"
                             f"   - {rel_wd} not found ('session_dir' variable as relative path from the current dir;"
                             f"   - {rel_sd} not found ('session_dir' variable as relative path from the script dir.")


def create_dir(session_dir: str,
               session_name: str) -> str:
    """
    Create a new directory of the given name in the given directory.
    If it already exists, add a unique identifier at the end of the directory name.

    :param session_dir: Path where to create the directory.
    :param session_name: Name of the directory to create.
    :return: Path to the newly created directory.
    """

    if isdir(join(session_dir, session_name)):
        print(f"Directory conflict: you are going to overwrite {join(session_dir, session_name)}.")
        # Find all the duplicated folders
        session_name += '_'
        copies = [folder for folder in listdir(session_dir) if isdir(join(session_dir, folder)) and
                  folder.find(session_name) == 0]
        # Get the indices of copies
        indices = [int(folder[len(session_name):]) for folder in copies]
        # The new copy is the max index + 1
        max_idx = max(indices) if len(indices) > 0 else 0
        session_name += f'{max_idx + 1}'

    session = join(session_dir, session_name)
    print(f"Create a new directory {session} for this session.")
    makedirs(session)
    return session


def copy_dir(src_dir: str,
             dest_dir: str,
             dest_name: Optional[str] = None,
             sub_folders: Optional[str] = None) -> str:
    """
    Copy the source directory to the destination directory.

    :param src_dir: Source directory to copy.
    :param dest_dir: Parent of the destination directory to copy.
    :param dest_name: Destination directory to copy to.
    :param sub_folders: If sub folders are specified, the latest is actually copied.
    :return: Path to the newly copied directory.
    """

    if dest_name is not None and isdir(join(dest_dir, dest_name)):
        print(f"Directory conflict: you are going to overwrite {join(dest_dir, dest_name)}.")
        # Find all the duplicated folders
        dest_name += '_'
        copies = [folder for folder in listdir(dest_dir) if isdir(join(dest_dir, folder)) and
                  folder.find(dest_name) == 0]
        # Get the indices of the copies
        indices = [int(folder[len(dest_name):]) for folder in copies]
        # The new copy is the max index + 1
        max_id = max(indices) if len(indices) > 0 else 0
        dest_name += f'{max_id + 1}'

    dest = join(dest_dir, dest_name) if dest_name is not None else dest_dir
    print(f"Copying the source directory {src_dir} to {dest} for this session.")
    if sub_folders is None:
        copytree(src_dir, dest)
    else:
        copytree(join(src_dir, sub_folders), join(dest, sub_folders))
    return dest
