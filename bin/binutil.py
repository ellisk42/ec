import os
import sys


# Add the root directory to sys.path in order to correctly import files from the 'lib'
# python package.
#
# To ensure this code runs, add 'import binutil' to the top of each file.
#
# For more info, see: https://stackoverflow.com/a/6466139/2573242
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
