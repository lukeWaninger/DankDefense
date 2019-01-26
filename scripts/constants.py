import os
from pathlib import Path

BUCKET = 'dank-defense'
FEATURES_KEY = 'features'

try:
    with open(os.path.join(Path.home(), 'AWS_KEY')) as f:
        AWS_KEY = f.read().strip()

    with open(os.path.join(Path.home(), 'AWS_SECRET')) as f:
        AWS_SECRET = f.read().strip()
except FileNotFoundError:
    pass