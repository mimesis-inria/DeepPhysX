import os
import sys
from distutils.util import strtobool
from importlib.util import find_spec

PACKAGES = {'sofa': True,
            'torch': True}
GIT = {'sofa': '',
       'torch': ''}


if __name__ == '__main__':

    # Check passed options
    for option in sys.argv[1:]:

        # Check option format
        if len(option.split('=')) != 2 or False in [len(split_opt) != 0 and ' ' not in split_opt
                                                    for split_opt in option.split('=')]:
            print(f"WARNING: Invalid option format with '{option}'."
                  f"\nMust be <package_name>=<do_install> with <package_name> in {list(PACKAGES.keys())} and "
                  f"<do_install> being either True (1) or False (0).")
            quit(0)

        # Check option name
        if option.split('=')[0].lower() not in PACKAGES:
            print(f"WARNING: Invalid option name with '{option}'."
                  f"\nMust be in {list(PACKAGES.keys())}.")
            quit(0)

        # Check option value
        if option.split('=')[1] not in ['0', '1', 'False', 'True']:
            print(f"WARNING: Invalid option value with '{option}'."
                  f"\nMust be either True (1) or False (0).")
            quit(0)

    # If packages are in current repository, add theme by default
    standard_name = {'sofa': 'Sofa', 'torch': 'Torch'}
    for lower, standard in standard_name.items():
        if not PACKAGES[lower]:
            PACKAGES[lower] = os.path.exists(standard)

    # Configure packages values
    for option in sys.argv[1:]:
        name, value = option.split('=')
        PACKAGES[name.lower()] = bool(int(value)) if len(value) == 1 else bool(strtobool(value))

    # Clone missing packages
    for name, value in PACKAGES.items():
        if value and find_spec(standard_name[name]) is None:
            print(f'Package {standard_name[name]} not found, cloning from {GIT[name]}...')
            os.system(f'git clone {GIT[name]} {standard_name[name]}')

    # Get PACKAGE variable position in current file
    with open(__file__, 'r') as f:
        datafile = f.readlines()
        for i, line in enumerate(datafile):
            if 'PACKAGES' in line:
                beginning = i
                break
        for i, line in enumerate(datafile[beginning:]):
            if '}' in line:
                end = beginning + i
                break

    # Replace with new value
    new_package = [f"'{str(key)}': {value},\n" for key, value in PACKAGES.items()]
    new_package[0] = 'PACKAGES = {' + new_package[0]
    for i in range(1, len(new_package)):
        new_package[i] = ' ' * len('PACKAGES = {') + new_package[i]
    new_package[-1] = new_package[-1].replace(',', '}')
    with open(__file__, 'w') as f:
        f.writelines(datafile[:beginning])
        f.writelines(new_package)
        f.writelines(datafile[end + 1:])

    # Display new config
    print("Applied config with values:"
          "\n\tPACKAGE_CORE: True")
    for name, value in PACKAGES.items():
        print(f"\tPACKAGE_{name.upper()}: {value}")
