from os import listdir, symlink, unlink, mkdir
from os.path import join, islink, abspath, pardir, isdir
from pathlib import Path
from sys import argv
from site import USER_SITE
from shutil import rmtree

from config import check_repositories


# Check user entry
if len(argv) != 2 or argv[1] not in ['set', 'del']:
    print("\nInvalid script option."
          "\nRun 'python3 dev.py set' to link DPX to your site package."
          "\nRun 'python3 dev.py del' to remove DPX links in your site package.")
    quit()

# Check repositories names
check_repositories()

# Init DeepPhysX packages and dependencies to install
PROJECT = 'DeepPhysX'
packages = ['Core']
available = ['Torch', 'Sofa']
root = abspath(join(Path(__file__).parent.absolute(), pardir))

# Option 1: create the symbolic links
if argv[1] == 'set':

    # Create main repository in site-packages
    if not isdir(join(USER_SITE, PROJECT)):
        mkdir(join(USER_SITE, PROJECT))

    # Link to every existing packages
    for package_name in listdir(root):
        if package_name in available:
            packages.append(package_name)

    # Create symbolic links in site-packages
    for package_name in packages:
        if not islink(join(USER_SITE, PROJECT, package_name)):
            symlink(src=join(root, package_name, 'src'), dst=join(USER_SITE, PROJECT, package_name))
            print(f"Linked {join(USER_SITE, PROJECT, package_name)} -> {join(root, package_name, 'src')}")

# Option 2: remove the symbolic links
else:

    if isdir(join(USER_SITE, PROJECT)):
        for package_name in listdir(join(USER_SITE, PROJECT)):
            unlink(join(USER_SITE, PROJECT, package_name))
            print(f"Unlinked {join(USER_SITE, PROJECT, package_name)} -> {join(root, package_name, 'src')}")
        rmtree(join(USER_SITE, PROJECT))
