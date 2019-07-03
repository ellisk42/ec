"""
The below `sys.path.append` code will add the root directory to sys.path in order to
correctly import files from the 'dreamcoder' Python package.

Additionally, some bin scripts depend on the pregex code.

This sys.path manipulation is required in order to preserve the same style of running
commands as has been used in the past for this project (mainly, running most experiments
from a large array of scripts that depend on methods and classes defined under other
modules now in the 'dreamcoder' package). A better approach would be to have a single entry
point Python script at the root of the repository which ensures that imports between
sibling folders work correctly.

To ensure this sys.path update runs, add `import binutil` to the top of each file.

For more info, see: https://stackoverflow.com/a/6466139/2573242
"""

import os
import sys

repo_root = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, 'pregex'))
