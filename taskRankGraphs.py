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
plot.style.use('seaborn-whitegrid')

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
	metricsToPlot,
	export=None):
	"""Plot times vs. the desired metrics for each iteration."""
	for j, path in enumerate(resultPaths):
		result = loadfun(path)
		print("loaded path:", path)
		if not hasattr(result, "recognitionTaskMetrics"):
			print("No recognitionTaskMetrics found, aborting.")
			assert False

		iterations = result.parameters['iterations']
		recognitionTaskMetrics = result.recognitionTaskMetrics

		for t in recognitionTaskMetrics:
			print(t.name)

		# Get all the times.
		taskTimes = [recognitionTaskMetrics[t]['recognitionBestTimes'] for t in recognitionTaskMetrics]
		# Replace the Nones with -1 for the purpose of this.
		taskTimes = [time if time is not None else -1.0 for time in taskTimes]

		for k, metricToPlot in enumerate(metricsToPlot):
			print("Plotting metric: " + metricToPlot)
			taskMetrics = [recognitionTaskMetrics[t][metricToPlot] for t in recognitionTaskMetrics]
			plot.scatter(taskTimes, taskMetrics)
			plot.xlabel('Recognition Best Times')
			plot.ylabel(metricToPlot)
			plot.savefig(os.path.join(export, metricToPlot + "_iters_" + str(iterations) + ".png"))


	
if __name__ == "__main__":
	import sys

	import argparse
	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--checkpoints",nargs='+')
	parser.add_argument("--metricsToPlot", nargs='+',type=str)
	parser.add_argument("--export","-e",
						type=str, default='data')

	arguments = parser.parse_args()

	plotTimeMetrics(arguments.checkpoints,
					arguments.metricsToPlot,
					arguments.export)