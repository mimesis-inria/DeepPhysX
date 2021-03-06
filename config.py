from os import chdir, pardir, system, sep, rename, listdir, getcwd, mkdir
from os.path import exists, abspath, join
from json import dump
from shutil import move

PROJECT = 'DeepPhysX'
PACKAGES = {'Torch': False,
            'Sofa': False}
AVAILABLE = {'AI': ['Torch'],
             'Simulation': ['Sofa']}
GIT = {'Torch': 'https://github.com/mimesis-inria/DeepPhysX.Torch.git',
       'Sofa': 'https://github.com/mimesis-inria/DeepPhysX.Sofa.git'}
ANSWERS = ['y', 'yes', 'n', 'no']


def check_repositories():

    # Check current repository
    path = abspath(join(__file__, pardir, pardir))
    repository = abspath(join(__file__, pardir)).split(sep)[-1]
    chdir(join(path, repository))
    size = 2
    if repository != 'Core':
        print(f"WARNING: Wrong repository, moving '{repository}' --> '{join(PROJECT, 'Core')}'")
        chdir(pardir)
        rename(repository, 'Core')
        mkdir(PROJECT)
        move(src=join(path, f'Core{sep}'),
             dst=join(path, PROJECT))
        size += 1
        path = join(path, PROJECT)
        chdir(join(path, 'Core'))

    # Check other repositories
    for i in range(size):
        for repository in listdir(getcwd()):
            if 'DeepPhysX.' in repository:
                if repository[10:] in PACKAGES.keys():
                    print(f"WARNING: Wrong repository, moving '{repository}' --> '{join(PROJECT, repository[10:])}'")
                    rename(repository, repository[10:])
                    if not exists(join(path, f'{repository[10:]}{sep}')):
                        move(src=join(getcwd(), f'{repository[10:]}{sep}'),
                             dst=path)
        chdir(pardir)
    chdir(join(path, 'Core'))


if __name__ == '__main__':

    # Check repositories names
    check_repositories()

    # Get user entry for each package
    for package_type, package_names in AVAILABLE.items():
        print(f"\nAvailable {package_type} packages : {package_names}")
        for package_name in package_names:
            while (do_install := input(f" >> Installing package {package_name} (y/n): ").lower()) not in ANSWERS:
                pass
            PACKAGES[package_name] = do_install in ANSWERS[:2]

    # Ask user confirmation
    print("\nApplying following configuration: \n  * DeepPhysX.Core: True (default)")
    for package_name, do_install in PACKAGES.items():
        print(f"  * DeepPhysX.{package_name}: {do_install}")
    while (do_validate := input("Confirm (y/n): ")) not in ANSWERS:
        pass
    if do_validate in ANSWERS[2:]:
        print("Aborting")
        quit()

    # Save config
    with open('config.json', 'w') as file:
        dump(PACKAGES, file)
    print("Configuration saved in 'config.json'")

    # Clone missing packages
    chdir(pardir)
    for package_name, do_install in PACKAGES.items():
        if do_install and not exists(package_name):
            print(f"\nPackage {package_name} not found, cloning from {GIT[package_name]}")
            system(f'git clone {GIT[package_name]} {package_name}')

    # End config
    print("\nConfiguration done, install DeepPhysX with the following commands:"
          "\n  - 'pip install .'       for user mode"
          "\n  - 'python3 dev.py set'  for developer mode")
