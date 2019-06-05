"""
Puddleworld primitives.

Converts a PyCCG ontology into primitives for Dreamcoder.
"""
import sys
sys.path.insert(0, "pyccg")
sys.path.insert(0, "pyccg/nltk")
from pyccg.logic import TypeSystem, Ontology, Expression
from puddleworld.ontology import puddleworldOntology

from type import baseType
from utilities import eprint


"""
tLayoutMap = baseType('layoutMap')
tObjectMap = baseType('ObjectMap')
tLocation= baseType('location')

primitives = []
"""

def loadPuddleworldOntology():
	return []

if __name__ == "__main__":
	primitives = loadPuddleworldOntology()