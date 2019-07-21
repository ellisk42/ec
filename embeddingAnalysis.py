"""
Recognition model embedding analyses for CogSci 2019.
Self-contained version of the analyses largely found at taskRankGraphs.py
Performs the following analyses:
	1. Embedding entropy vs. solve times for heldout tasks.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from scipy.stats import entropy

from taskRankGraphs import loadResult


def load_checkpoint_metrics(checkpoints, iters):
	"""
	Loads checkpoints for iterations in iters.
	:ret: dict {iter : list of recognitionTaskMetric dicts}
	"""
	results = {i : [] for i in iters}
	for c in checkpoints:
		if os.path.isfile(c):
			result, domain, iterations, recognitionTaskMetrics = loadResult(c, export=None)
			if int(iterations) in results:
				results[iterations].append(recognitionTaskMetrics)
	return results

def grammar_entropy_analysis(metric_dicts, export):
	"""
	For results at each iteration:
	Unigram, bigram, and log production embedding entropies for solved tasks,
	and for solved vs. unsolved tasks.
	"""
	plots = {it : {
		'bigram_solved' : [],
		'unigram_solved' : [],

	} for it in metric_dicts}
	for it in metric_dicts:
		for metric_dict in metric_dicts[it]:
			heldout_tasks = [t for t in metric_dict if 'heldoutTaskLogProductions' in metric_dict[t]]
			solved = [t for t in heldout_tasks if metric_dict[t]['heldoutTestingTimes'] is not None]
			unsolved = [t for t in heldout_tasks if t not in solved]

			print("Heldout %d, solved %d, unsolved %d" % (len(heldout_tasks), len(solved), len(unsolved)))

			# Get all the solve times for the solved tasks
			solve_times = [metric_dict[t]['heldoutTestingTimes'] for t in solved]
			# Sort by solve time.
			solve_times, solved = zip(*sorted(zip(solve_times, solved), key=lambda t:t[0]))
			bigram_solved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][1].flatten()) for t in solved]
			unigram_solved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][0].flatten()) for t in solved]

			bigram_unsolved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][1].flatten()) for t in unsolved]
			unigram_unsolved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][0].flatten()) for t in unsolved]
			
			print("Average entropy for solved: unigram: %f, bigram: %f" % (np.mean(unigram_solved_entropy), np.mean(bigram_solved_entropy)))
			print("Average entropy for unsolved: unigram: %f, bigram: %f" % (np.mean(unigram_unsolved_entropy), np.mean(bigram_unsolved_entropy)))

			print("Median entropy for solved: unigram: %f, bigram: %f" % (np.median(unigram_solved_entropy), np.median(bigram_solved_entropy)))
			print("Median entropy for unsolved: unigram: %f, bigram: %f" % (np.median(unigram_unsolved_entropy), np.median(bigram_unsolved_entropy)))
			
			print("Average solve time %f, median solve time: %f" % (np.mean(solve_times), np.median(solve_times)))

			plots[it]['bigram_solved'].append((solve_times, bigram_solved_entropy))
			plots[it]['unigram_solved'].append((solve_times, unigram_solved_entropy))

	# Plots.
	def seaborn_regplots(data, key, export_dir, export_tag):
		import os

		fig = plt.figure(figsize=(20, 10))
		for i, (xs, ys) in enumerate(data[key]):
			sns.regplot(x=xs, y=ys, label=str(i))

		export_title = '%s_%s.png' % (export_tag, key)
		fig.savefig(os.path.join(export_dir, export_title))
	for iter in plots:
		seaborn_regplots(plots[iter], 'bigram_solved', export_dir, str(iter))
		seaborn_regplots(plots[iter], 'unigram_solved', export_dir, str(iter))
		


if __name__ == "__main__":
	import sys
	import argparse

	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--checkpoints",nargs='+')
	parser.add_argument("--iters",nargs='+',type=int)
	parser.add_argument("--export_dir", type=str)

	args = parser.parse_args()

	# Load all the metric_dicts for all of the checkpoints.
	metric_dicts = load_checkpoint_metrics(args.checkpoints, args.iters)

	# Unigram, bigram, and log production embedding entropies for solved tasks,
	# and for solved vs. unsolved tasks.
	if True:
		grammar_entropy_analysis(metric_dicts, args.export_dir)

