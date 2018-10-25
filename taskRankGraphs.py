"""Creates graphs for the task reranking metrics."""

from ec import *
import dill
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

import matplotlib

import text
from text import LearnedFeatureExtractor

def loadfun(x):
	with open(x, 'rb') as handle:
		result = dill.load(handle)
	return result

class Bunch(object):
	def __init__(self, d):
		self.__dict__.update(d)

	def __setitem__(self, key, item):
		self.__dict__[key] = item

	def __getitem__(self, key):
		return self.__dict__[key]

relu = 'relu'
tanh = 'tanh'
sigmoid = 'sigmoid'
DeepFeatureExtractor = 'DeepFeatureExtractor'
LearnedFeatureExtractor = 'LearnedFeatureExtractor'
TowerFeatureExtractor = 'TowerFeatureExtractor'

def parseResultsPath(p):
	def maybe_eval(s):
		try:
			return eval(s)
		except BaseException:
			return s

	p = p[:p.rfind('.')]
	domain = p[p.rindex('/') + 1: p.index('_')]
	rest = p.split('_')[1:]
	if rest[-1] == "baselines":
		rest.pop()
	parameters = {ECResult.parameterOfAbbreviation(k): maybe_eval(v)
				  for binding in rest if '=' in binding
				  for [k, v] in [binding.split('=')]}
	parameters['domain'] = domain
	return Bunch(parameters)


def plotTimeMetrics(
	resultPaths,
	export=None):
	"""Plot times vs. the desired metrics for each iteration."""
	for j, path in enumerate(resultPaths):
		result = loadfun(path)
		print("loaded path:", path)

		if hasattr(result, "recognitionTaskTimes") and result.recognitionTaskTimes:
			print("Has recognitionTaskTimes")
			for task in result.recognitionTaskTimes:
				print(result.recognitionTaskTimes[task])

	
if __name__ == "__main__":
	import sys

	import argparse
	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--checkpoints",nargs='+')
	parser.add_argument("--export","-e",
						type=str, default=None)

	arguments = parser.parse_args()

	plotTimeMetrics(arguments.checkpoints,
					arguments.export)