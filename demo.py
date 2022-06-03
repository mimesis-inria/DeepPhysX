import os
import sys
import argparse


parser = argparse.ArgumentParser(prog='demo', description='Specify the name of the demo.')
parser.add_argument('Demo', metavar='demo', type=str, help='Name of the demo to run')
args = parser.parse_args()

demo = args.Demo
demos = ['armadillo', 'beam', 'liver']

if demo not in demos:
    raise ValueError(f"Unknown demo '{demo}', available are: {demos}")

repo = os.path.join(os.path.dirname(__file__), 'examples', 'demos', demo[0].upper() + demo[1:].lower(), 'FC')
os.chdir(repo)
os.system(f'{sys.executable} interactive.py')
