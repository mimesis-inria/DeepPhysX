from os import listdir, symlink, unlink
from os.path import dirname, join, islink
from pathlib import Path
from sys import argv
from site import USER_SITE
from config import PACKAGES


# Init DeepPhysX packages and dependencies to install
roots = ['Core']
available = {'ai': ['Torch'],
             'simu': ['Sofa']}
path = Path(__file__).parent.absolute()

# Include user config
user_ai_packages = []
user_simu_packages = []
for user_packages, key in zip([user_ai_packages, user_simu_packages], ['ai', 'simu']):
    for root in available[key]:
        if PACKAGES[root.lower()]:
            user_packages.append(root)

# Define the main packages to install
roots += user_ai_packages
roots += user_simu_packages

# Create symbolic link in user site
if len(argv) == 1:
    for root in roots:
        dpx_package = [f for f in listdir(join(path, root)) if 'DeepPhysX' in f][0]
        if not islink(join(USER_SITE, dpx_package)):
            symlink(src=join(path, root, dpx_package), dst=join(USER_SITE, dpx_package))
            print(f'Linked {join(USER_SITE, dpx_package)} -> {join(path, root, dpx_package)}')

# Remove symbolic links
elif argv[1] == '-u':
    dpx_packages = [f for f in listdir(USER_SITE) if 'DeepPhysX' in f and islink(join(USER_SITE, f))]
    for dpx in dpx_packages:
        unlink(join(USER_SITE, dpx))

else:
    print("Invalid script option. "
          "Run 'python3 dev_install.py' to link DPX to your site package. "
          "Run 'python3 dev_install.py -u' to remove DPX links in your site package.")
