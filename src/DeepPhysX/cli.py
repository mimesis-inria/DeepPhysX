from argparse import ArgumentParser
from json import load
from os import listdir, getcwd, readlink, chdir
from os.path import islink, dirname, join, abspath, isdir
from platform import system
from shutil import copytree, rmtree
from subprocess import run
from sys import executable


def is_pip_installed():

    import DeepPhysX.Core
    return not islink(DeepPhysX.Core.__path__[0])


def get_sources():

    import DeepPhysX
    site_packages = dirname(DeepPhysX.__path__[0])
    metadata_repo = [f for f in listdir(site_packages) if f.split('-')[0] == 'DeepPhysX' and
                     ('.dist-info' in f or '.egg-info' in f)]
    if len(metadata_repo) == 0:
        quit(print("The project does not seem to be properly installed. Try to re-install using 'pip'."))
    elif len(metadata_repo) > 1:
        quit(print("There might be several version of the project, try to clean your site-packages."))
    metadata_repo = metadata_repo.pop(0)
    if 'direct_url.json' not in listdir(join(site_packages, metadata_repo)):
        return None
    with open(join(site_packages, metadata_repo, 'direct_url.json'), 'r') as file:
        direct_url = load(file)
    if system() == 'Linux':
        return abspath(direct_url['url'].split('//')[1])
    elif system() == 'Windows':
        return abspath(direct_url['url'].split('///')[1])
    else:
        return abspath(direct_url['url'].split('///')[1])


def copy_examples_dir():

    user = input(f"WARNING: The project was installed with pip, examples must be run in a new repository to avoid "
                 f"writing data in your installation of SSD. Allow the creation of this new repository "
                 f"'{join(getcwd(), 'DPX_examples')}' to run examples (use 'DPX --clean' to cleanly"
                 f"remove it afterward) (y/n):")
    if user.lower() not in ['y', 'yes']:
        quit(print("Aborting."))

    import DeepPhysX.examples
    copytree(src=DeepPhysX.examples.__path__[0],
             dst=join(getcwd(), 'DPX_examples'))


def clean_examples_dir():

    if not isdir(examples_dir := join(getcwd(), 'DPX_examples')):
        quit(print(f"The directory '{examples_dir}' does not exists."))
    user = input(f"Do you want to remove the repository '{examples_dir}' (y/n):")
    if user.lower() not in ['y', 'yes']:
        quit(print("Aborting."))
    rmtree(examples_dir)


def print_available_examples(examples):

    example_names = sorted(list(examples.keys()))
    example_per_repo = {}
    for example_name in example_names:
        if type(examples[example_name]) == str:
            root, repo = examples[example_name].split('.')[0], examples[example_name].split('.')[1]
        else:
            root, repo = examples[example_name][0].split('.')[0], examples[example_name][0].split('.')[1]
        repo = 'rendering' if repo == 'rendering-offscreen' else repo
        if root not in example_per_repo:
            example_per_repo[root] = {}
        if repo not in example_per_repo[root]:
            example_per_repo[root][repo] = []
        example_per_repo[root][repo].append(example_name)

    description = '\navailable examples:'
    for repo, sub_repos in example_per_repo.items():
        for sub_repo, names in sub_repos.items():
            description += f'\n   {repo}.{sub_repo}: {names}'
    print(description)


def execute_cli():

    description = "Command Line Interface dedicated to DPX examples."
    parser = ArgumentParser(prog='SSD', description=description)
    parser.add_argument('-c', '--clean', help='clean the example repository.', action='store_true')
    parser.add_argument('-g', '--get', help='get the full example repository locally.', action='store_true')
    parser.add_argument('-r', '--run', type=str, help='run one of the demo sessions.', metavar='')
    args = parser.parse_args()

    # Get a copy of the example repository if pip installed from PyPi.org
    if args.get:
        # Installed with setup_dev.py
        if not is_pip_installed():
            quit(print("The project was installed from sources in dev mode, examples will then be run in "
                       "'DeepPhysX.<pkg_name>.examples'."))
        # Installed with pip from sources
        if (source_dir := get_sources()) is not None:
            quit(print(f"The project was installed with pip from sources, examples will then be run in "
                       f"'{join(source_dir, 'examples')}'."))
        # Installed with pip from PyPi
        copy_examples_dir()
        return

    # Clean the examples repository if pip installed from PyPi.org
    elif args.clean:
        # Installed with setup_dev.py
        if not is_pip_installed():
            quit(print("The project was installed from sources in dev mode, you cannot clean "
                       "'DPX.<pck_name>.examples'."))
        # Installed with pip from sources
        if (source_dir := get_sources()) is not None:
            quit(print(f"The project was installed with pip from sources, you cannot clean "
                       f"'{join(source_dir, 'examples')}'."))
        # Installed with pip from PyPi
        clean_examples_dir()
        return

    examples = {'armadillo': 'Core/demos/Armadillo/FC/interactive.py',
                'beam': 'Core/demos/Beam/FC/interactive.py',
                'liver': 'Core/demos/Liver/FC/interactive.py'}

    # Run a demo script
    if (example := args.run) is not None:
        # Check the example name
        if example.lower() not in examples.keys():
            print(f"Unknown demo '{example}'.")
            quit(print_available_examples(examples))
        # Get the example directory
        if not is_pip_installed():
            import DeepPhysX.Core
            source_dir = readlink(DeepPhysX.Core.__path__[0])
            examples_dir = join(dirname(dirname(source_dir)), 'examples')
            repo = join(*examples[example].split('/')[1:-1])
        elif (source_dir := get_sources()) is not None:
            examples_dir = join(source_dir, 'examples')
            repo = join(*examples[example].split('/')[1:-1])
        else:
            if not isdir(join(getcwd(), 'DPX_examples')):
                print(f"The directory '{join(getcwd(), 'DPX_examples')}' does not exists.")
                copy_examples_dir()
            examples_dir = join(getcwd(), 'DPX_examples')
            repo = join(*examples[example].split('/')[:-1])
        # Run the example
        script = examples[example].split('/')[-1]
        chdir(join(examples_dir, repo))
        run([f'{executable}', f'{script}'], cwd=join(examples_dir, repo))

        return

    # No command
    else:
        parser.print_help()
        print_available_examples(examples)


if __name__ == '__main__':
    execute_cli()
