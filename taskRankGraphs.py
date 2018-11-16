"""
Creates graphs for task re-ranking metrics from an ECResults checkpoint.
Requires metrics to be available in a recognitionTaskMetrics dict: you can specify this via --storeTaskMetrics.
Or you can attempt to back add them using the --addTaskMetrics function.

Usage: Example script is in taskRankGraphs.

Note: this requires a container with sklearn installed. A sample container is available in /om2/user/zyzzyva/ec/sklearn-container.img
"""

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
from scipy import stats
from scipy.stats import entropy

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

def loadResult(path, export):
	result = loadfun(path)
	print("loaded path:", path)
	if not hasattr(result, "recognitionTaskMetrics"):
		print("No recognitionTaskMetrics found, aborting.")
		assert False

	domain = parseResultsPath(path)['domain']
	iterations = result.parameters['iterations']
	recognitionTaskMetrics = result.recognitionTaskMetrics

	# Create a folder for the domain if it does not exist.
	if not os.path.exists(os.path.join(export, domain)):
		os.makedirs(os.path.join(export, domain))
		
	return result, domain, iterations, recognitionTaskMetrics

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))

def normalizeAndEntropy(x):
	return entropy(softmax(x))

def exportTaskTimes(
	resultPaths,
	timesArg,
	export):
	"""Exports the task time information to the output file"""
	for j, path in enumerate(resultPaths):
		print("Logging result: " + path)
		result, domain, iterations, recognitionTaskMetrics = loadResult(path, export)
		enumerationTimeout = result.parameters['enumerationTimeout'] 

		# Get all the times.
		tasks = [t for t in recognitionTaskMetrics if timesArg in recognitionTaskMetrics[t]]
		taskTimes = [recognitionTaskMetrics[t][timesArg] for t in tasks]

		solvedTaskTimes = [time for time in taskTimes if time is not None ]
		# Replace the Nones with enumeration timeout for the purpose of this.
		taskTimes = [time if time is not None else enumerationTimeout for time in taskTimes]
		taskNames=[t.name for t in tasks]


		# Log Results.
		print("Solved tasks: " + str(len(solvedTaskTimes)))
		print ("Total tasks this round: " + str(len(taskTimes)))
		print("Average solve time (secs): " + str(np.mean(solvedTaskTimes)))
		print("Total time spent solving tasks (secs): " + str(np.sum(taskTimes)))

		for i, name in enumerate(taskNames):
			print("TASK: " + name + " TIME: " + str(taskTimes[i]))


def plotTimeMetrics(
	resultPaths,
	outlierThreshold,
	metricsToPlot,
	timesArg,
	export=None):
	"""Plots task times vs. the desired metrics (ie. entropy) for each checkpoint iteration."""


	for j, path in enumerate(resultPaths):
		result, domain, iterations, recognitionTaskMetrics = loadResult(path, export)

		# Get all the times.
		tasks = [t for t in recognitionTaskMetrics if timesArg in recognitionTaskMetrics[t]]
		taskTimes = [recognitionTaskMetrics[t][timesArg] for t in tasks]
		# Replace the Nones with -1 for the purpose of this.
		taskTimes = [time if time is not None else -1.0 for time in taskTimes]
		taskNames=[t.name for t in tasks]

		for k, metricToPlot in enumerate(metricsToPlot):
			print("Plotting metric: " + metricToPlot)

			taskMetrics = [recognitionTaskMetrics[t][metricToPlot] for t in tasks]

			if outlierThreshold:
				# Threshold to only outlierThreshold stddeviations from the median
				ceiling = outlierThreshold
				noOutliersNames, noOutliersTimes, noOutliersMetrics = [], [], []
				for t in range(len(taskTimes)):
					if taskTimes[t] < ceiling:
						noOutliersTimes.append(taskTimes[t])
						noOutliersMetrics.append(taskMetrics[t])
						noOutliersNames.append(taskNames[t])
				taskNames, taskTimes, taskMetrics = noOutliersNames, noOutliersTimes, noOutliersMetrics

			if outlierThreshold:
				xlabel = ('Recognition Best Times, Outlier Threshold: %d' % (outlierThreshold))
			else:
				xlabel = ('Recognition Best Times')
			title = ("Domain: %s, Iteration: %d" % (domain, iterations))
			ylabel = metricToPlot
			export_name = metricToPlot + "_iters_" + str(iterations) + "outlier_threshold_" + str(outlierThreshold) + "_time_plot.png"

			plot.scatter(taskTimes, taskMetrics)
			plot.xlabel(xlabel)
			plot.ylabel(ylabel)
			plot.title(title)
			plot.savefig(os.path.join(export, domain, export_name))

			print("Plotted metric without labels.")

			# Also try plotting with labels.
			times_and_metrics = np.column_stack((taskTimes, taskMetrics))
			plotEmbeddingWithLabels(
				times_and_metrics,
				taskNames,
				title,
				os.path.join(export, domain, "labels_" + export_name),
				xlabel,
				ylabel)

			print("Plotted metric with labels.")
			return



def plotEmbeddingWithLabels(embeddings, labels, title, exportPath, xlabel=None, ylabel=None):
	plot.figure(figsize=(20,20))
	for i, label in enumerate(labels):
		x, y = embeddings[i, 0], embeddings[i, 1]
		plot.scatter(x,y)
		plot.text(x+0.02, y+0.02, label, fontsize=8)
	plot.title(title)
	if xlabel:
		plot.xlabel(xlabel)
	if ylabel:
		plot.ylabel(ylabel)
	plot.savefig(exportPath)
	return


def plotTSNE(
	resultPaths,
	metricsToCluster,
	export=None):
	"""Plots TSNE clusters of the given metrics. Requires Sklearn."""

	from sklearn.manifold import TSNE

	if metricsToCluster is None:
		return

	for j, path in enumerate(resultPaths):
		result, domain, iterations, recognitionTaskMetrics = loadResult(path, export)

		for k, metricToCluster in enumerate(metricsToCluster):
			print("Clustering metric: " + metricToCluster)
			tsne = TSNE(random_state=0)
			taskNames, taskMetrics = [], []

			print(len(recognitionTaskMetrics))

			for task in recognitionTaskMetrics:
				if metricToCluster in recognitionTaskMetrics[task] and recognitionTaskMetrics[task][metricToCluster] is not None:
					taskNames.append(task.name)  
					taskMetrics.append(recognitionTaskMetrics[task][metricToCluster])
			taskNames = np.array(taskNames)
			taskMetrics = np.array(taskMetrics)
			print(taskNames.shape, taskMetrics.shape)
			print("Clustering %d tasks with embeddings of shape: %s" % (len(taskMetrics), str(taskMetrics[0].shape)) )
			
			clusteredTaskMetrics = tsne.fit_transform(taskMetrics)
			title = ("Metric: %s, Domain: %s, Iteration: %d" % (metricToCluster, domain, iterations))
			plotEmbeddingWithLabels(clusteredTaskMetrics, 
				taskNames, 
				title, 
				os.path.join(export, domain, metricToCluster + "_iters_" + str(iterations) + "_tsne_default.png"))

	
if __name__ == "__main__":
	import sys

	import argparse
	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--checkpoints",nargs='+')
	parser.add_argument("--metricsToPlot", nargs='+',type=str)
	parser.add_argument("--times", type=str, default='recognitionBestTimes')
	parser.add_argument("--exportTaskTimes", type=bool)
	parser.add_argument("--outlierThreshold", type=float, default=None)
	parser.add_argument("--metricsToCluster", nargs='+', type=str, default=None)
	parser.add_argument("--export","-e",
						type=str, default='data')

	arguments = parser.parse_args()

	if arguments.exportTaskTimes:
		exportTaskTimes(arguments.checkpoints,
						arguments.times,
						arguments.export)

	if arguments.metricsToPlot:
		plotTimeMetrics(arguments.checkpoints,
						arguments.outlierThreshold,
						arguments.metricsToPlot,
						arguments.times,
						arguments.export)

	if arguments.metricsToCluster:
		plotTSNE(arguments.checkpoints,
				 arguments.metricsToCluster,
				 arguments.export)