from os import sep, chdir, mkdir, listdir, getcwd, rename, symlink, unlink, remove
from os.path import dirname, join, isfile, exists, isdir, islink
from pathlib import Path
from shutil import move, rmtree, which
from site import USER_SITE
from subprocess import run
from sys import argv

from pip._internal.operations.install.wheel import PipScriptMaker

PROJECT = 'DeepPhysX'
AI_PACKAGES = ['Torch']
SIMU_PACKAGES = ['Sofa']
PACKAGES = ['Core'] + AI_PACKAGES + SIMU_PACKAGES
GIT = {'Torch': 'https://github.com/mimesis-inria/DeepPhysX.Torch.git',
       'Sofa': 'https://github.com/mimesis-inria/DeepPhysX.Sofa.git'}


def check_repositories():

    dpx_path = dirname(dirname(__file__))

    # Check the DeepPhysX root
    if dpx_path.split(sep)[-1] != PROJECT:
        # Create the right root
        chdir(dpx_path)
        if dirname(__file__).split(sep)[-1] == PROJECT:
            rename(PROJECT, f'{PROJECT}.Core')
        mkdir(PROJECT)
        # Move DeepPhysX packages in the root
        for repository in listdir(getcwd()):
            if is_dpx_package(repository):
                move(src=join(dpx_path, repository),
                     dst=join(dpx_path, PROJECT))
        dpx_path = join(dpx_path, PROJECT)

    # Check the packages repositories
    for repository in listdir(dpx_path):
        # Core package
        if repository == PROJECT:
            rename(repository, 'Core')
        # Layers
        if 'DeepPhysX.' in repository and repository[10:] in PACKAGES:
            rename(src=join(dpx_path, repository),
                   dst=join(dpx_path, repository[10:]))

    return dpx_path


def is_dpx_package(repository):

    for key in ['DeepPhysX', 'DeepPhysX.'] + PACKAGES:
        if key in repository and isfile(join(repository, 'README.md')):
            with open(join(repository, 'README.md')) as f:
                if PROJECT in f.readline():
                    return True
    return False


def define_config(root_dir):

    # Get the user configuration
    config = ['Core']
    answers = ['y', 'yes', 'n', 'no']
    for package_list, package_type in zip([SIMU_PACKAGES, AI_PACKAGES], ['SIMULATION', 'AI']):
        print(f"\nAvailable {package_type} packages: {[f'{PROJECT}.{pkg}' for pkg in package_list]}")
        for pkg in package_list:
            while (user := input(f" >> Install package {f'{PROJECT}.{pkg}'} (y/n): ").lower()) not in answers:
                pass
            if user in answers[:2]:
                config.append(pkg)
    print(f"\nThe following packages will be installed : {[f'{PROJECT}.{package}' for package in config]}")
    while (user := input("Confirm (y/n): ").lower()) not in answers:
        pass
    if user in answers[2:]:
        quit(print("Aborting."))

    # Clone the missing packages
    chdir(root_dir)
    if len(config) > 1:
        for pkg in config[1:]:
            if not exists(pkg):
                print(f"\nPackage {f'{PROJECT}.{pkg}'} not found, cloning from {GIT[pkg]}...")
                run(['git', 'clone', f'{GIT[pkg]}', f'{pkg}'], cwd=root_dir)

    return config


if __name__ == '__main__':

    # Check the project tree
    root = check_repositories()

    # Check user entry
    if len(argv) == 2 and argv[1] not in ['set', 'del']:
        quit(print(f"\nInvalid script option."
                   f"\nRun 'python3 setup_dev.py set to link {PROJECT} to your site-packages folder."
                   f"\nRun 'python3 setup_dev.py del to remove {PROJECT} link from your site-packages folder."))

    # Option 1: create the symbolic links
    if len(argv) == 1 or argv[1] == 'set':

        # Get the user configuration
        packages = define_config(root)

        # Create the main repository in site-packages
        if not isdir(join(USER_SITE, PROJECT)):
            mkdir(join(USER_SITE, PROJECT))

        # Create symbolic links in site-packages
        for package in packages:
            if not islink(join(USER_SITE, PROJECT, package)):
                symlink(src=join(root, package, 'src', package),
                        dst=join(USER_SITE, PROJECT, package))
                print(f"\nLinked {join(USER_SITE, PROJECT, package)} -> {join(root, package)}")

        # Add examples and the CLI script
        if not isdir(join(USER_SITE, PROJECT, 'examples')):
            symlink(src=join(Path(__file__).parent.absolute(), 'examples'),
                    dst=join(USER_SITE, PROJECT, 'examples'))
            print(f"\nLinked {join(USER_SITE, PROJECT, 'examples')} -> {join(Path(__file__).parent.absolute(), 'examples')}")
        if not isfile(join(USER_SITE, PROJECT, 'cli.py')):
            symlink(src=join(Path(__file__).parent.absolute(), 'src', 'cli.py'),
                    dst=join(USER_SITE, PROJECT, 'cli.py'))

        # Create the CLI
        if which('DPX') is None:
            # Generate the scripts
            maker = PipScriptMaker(None, dirname(which('vedo')))
            generated_scripts = maker.make_multiple(['DPX = DeepPhysX.cli:execute_cli'])
            for script in generated_scripts:
                if script.split(sep)[-1].split('.')[0] != 'DPX':
                    remove(script)

    # Option 2: remove the symbolic links
    else:

        # Remove everything from site-packages
        if isdir(join(USER_SITE, PROJECT)):
            for package in listdir(join(USER_SITE, PROJECT)):
                if islink(join(USER_SITE, PROJECT, package)):
                    unlink(join(USER_SITE, PROJECT, package))
                    print(f"Unlinked {join(USER_SITE, PROJECT, package)} -> {join(root, package)}")
                elif isdir(join(USER_SITE, PROJECT, package)):
                    rmtree(join(USER_SITE, PROJECT, package))
                elif isfile(join(USER_SITE, PROJECT, package)):
                    remove(join(USER_SITE, PROJECT, package))
            rmtree(join(USER_SITE, PROJECT))

        # Remove the CLI
        if isfile(which('DPX')):
            remove(which('DPX'))
