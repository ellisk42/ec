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

from utilities import updateTaskSummaryMetrics

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

def add_solve_times(checkpoints, iters, metric_dicts):
	"""
	Method to get the solve times associated with the current stored recognition model.
	"""
	pass

def update_checkpoint_metrics(checkpoints, expert_iter, metric_dicts):
	updated_metrics = []
	for c in checkpoints:
		if os.path.isfile(c):
			result, domain, iterations, recognitionTaskMetrics = loadResult(c, export=None)
			if int(iterations) == expert_iter:
				expert_dict = recognitionTaskMetrics
				expert_grammar = result.grammars[-2]
				expert_heldout_tasks = [t for t in expert_dict if 'heldoutTaskLogProductions' in expert_dict[t]]
				expert_solved = [t for t in expert_heldout_tasks if expert_dict[t]['heldoutTestingTimes'] is not None]
				print("Found %d solved heldout tasks" % len(expert_solved))
				updateTaskSummaryMetrics(expert_dict, {
					expert_dict[t]['frontier'].task : expert_dict[t]['frontier'].expectedProductionUses(expert_grammar)
					for t in expert_solved
					},  'expectedProductionUses')
				updated_metrics.append(expert_dict)
	metric_dicts[expert_iter] = updated_metrics
	return metric_dicts




def grammar_entropy_analysis(metric_dicts, export_dir):
	"""
	For results at each iteration:
	Unigram, bigram, and log production embedding entropies for solved tasks,
	and for solved vs. unsolved tasks.
	"""
	from scipy.stats import pearsonr 

	plots = {it : {
		'bigram_solved' : [],
		'unigram_solved' : [],
		'embedding_solved' : []

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
			embedding_solved_entropy = [entropy(np.exp(metric_dict[t]['heldoutTaskLogProductions'].flatten())) for t in solved]
			embedding_unsolved_entropy = [entropy(np.exp(metric_dict[t]['heldoutTaskLogProductions'].flatten())) for t in unsolved]
			bigram_solved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][1].flatten()) for t in solved]
			unigram_solved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][0].flatten()) for t in solved]

			# bigram_unsolved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][1].flatten()) for t in unsolved]
			# unigram_unsolved_entropy = [entropy(metric_dict[t]['expectedProductionUsesMonteCarlo'][0].flatten()) for t in unsolved]
			
			# print("Average entropy for solved: unigram: %f, bigram: %f" % (np.mean(unigram_solved_entropy), np.mean(bigram_solved_entropy)))
			# print("Average entropy for unsolved: unigram: %f, bigram: %f" % (np.mean(unigram_unsolved_entropy), np.mean(bigram_unsolved_entropy)))

			# print("Median entropy for solved: unigram: %f, bigram: %f" % (np.median(unigram_solved_entropy), np.median(bigram_solved_entropy)))
			# print("Median entropy for unsolved: unigram: %f, bigram: %f" % (np.median(unigram_unsolved_entropy), np.median(bigram_unsolved_entropy)))

			print("Average entropy for solved: embedding: %f" % (np.mean(embedding_solved_entropy)))
			print("Average entropy for unsolved: embedding: %f" % (np.mean(embedding_unsolved_entropy)))

			print("Median entropy for solved:  embedding: %f" % (np.median(embedding_solved_entropy)))
			print("Median entropy for unsolved: embedding: %f" % (np.median(embedding_unsolved_entropy)))

			embedding_corr = pearsonr(embedding_solved_entropy, solve_times)[0]
			print("Embedding correlation with solve times: %f" % embedding_corr)

			embedding_corr = pearsonr(embedding_solved_entropy, bigram_solved_entropy)[0]
			print("Embedding correlation with bigram entropy: %f" % embedding_corr)
			
			# print("Average solve time %f, median solve time: %f" % (np.mean(solve_times), np.median(solve_times)))

			# solve_times, bigram_solved_entropy, unigram_solved_entropy = np.array(solve_times), np.array(bigram_solved_entropy), np.array(unigram_solved_entropy)
			# plots[it]['bigram_solved'].append((solve_times, bigram_solved_entropy))
			# plots[it]['unigram_solved'].append((solve_times, unigram_solved_entropy))
			# plots[it]['embedding_solved'].append((np.array(solve_times), embedding_solved_entropy))

	# Plots.
	def seaborn_regplots(data, key, export_dir, export_tag):
		import os

		fig = plt.figure(figsize=(20, 10))
		for i, (xs, ys) in enumerate(data[key]):
			sns.regplot(x=xs, y=ys, label=str(i))

		export_title = '%s_%s.png' % (export_tag, key)
		print("Exporting to ", os.path.join(export_dir, export_title))
		fig.savefig(os.path.join(export_dir, export_title))
	for iter in plots:
		seaborn_regplots(plots[iter], 'embedding_solved', export_dir, str(iter))
		# seaborn_regplots(plots[iter], 'bigram_solved', export_dir, str(iter))
		# seaborn_regplots(plots[iter], 'unigram_solved', export_dir, str(iter))


def grammar_distribution_similarity_analysis(metric_dicts, export_dir, domain):
	"""
	Explicitly expects the iters to be in the form (novice_iter, expert_iter)
	"""
	from sklearn.metrics.pairwise import cosine_similarity
	from scipy.stats import pearsonr 
	from collections import defaultdict

	iters = list(metric_dicts.keys())
	correlations = {
		'novice' : defaultdict(list),
		'expert' : defaultdict(list)
	}
	for n in range(len(metric_dicts[iters[0]])):
		print("Iterations: novice, %d, expert %d" % (iters[0], iters[1]))
		novice_dict, expert_dict = metric_dicts[iters[0]][n],  metric_dicts[iters[1]][n]
		expert_heldout_tasks = [t for t in expert_dict if 'heldoutTaskLogProductions' in expert_dict[t]]
		expert_solved = [t for t in expert_heldout_tasks if expert_dict[t]['heldoutTestingTimes'] is not None]
		novice_solved = [t for t in expert_heldout_tasks if novice_dict[t]['heldoutTestingTimes'] is not None]

		print("Novice solved: %d, expert solved %d" % (len(novice_solved), len(expert_solved)))

		# Compile a similarity matrix amongst the expert ground truths.
		expert_ground_truth = [expert_dict[t]['expectedProductionUses'] for t in expert_solved]
		expert_ground_truth_sims = cosine_similarity(np.array(expert_ground_truth)).flatten()

		expert_unigram, expert_bigram, expert_embedding = [expert_dict[t]['expectedProductionUsesMonteCarlo'][0].flatten() for t in expert_solved], \
															[expert_dict[t]['expectedProductionUsesMonteCarlo'][1].flatten() for t in expert_solved], \
															[expert_dict[t]['heldoutTaskLogProductions'].flatten() for t in expert_solved]
		novice_unigram, novice_bigram, novice_embedding = [novice_dict[t]['expectedProductionUsesMonteCarlo'][0].flatten() for t in expert_solved], \
															[novice_dict[t]['expectedProductionUsesMonteCarlo'][1].flatten() for t in expert_solved], \
															[novice_dict[t]['heldoutTaskLogProductions'].flatten() for t in expert_solved]
		
		expert_unigram_sims, expert_bigram_sims, expert_embedding_sims = cosine_similarity(np.array(expert_unigram)).flatten(), cosine_similarity(np.array(expert_bigram)).flatten(), cosine_similarity(np.array(expert_embedding)).flatten()
		novice_unigram_sims, novice_bigram_sims, novice_embedding_sims = cosine_similarity(np.array(novice_unigram)).flatten(), cosine_similarity(np.array(novice_bigram)).flatten(), cosine_similarity(np.array(novice_embedding)).flatten()

		novice_unigram_corr, novice_bigram_corr, novice_embedding_corr = pearsonr(expert_ground_truth_sims, novice_unigram_sims), pearsonr(expert_ground_truth_sims, novice_bigram_sims), pearsonr(expert_ground_truth_sims, novice_embedding_sims)
		expert_unigram_corr, expert_bigram_corr, expert_embedding_corr = pearsonr(expert_ground_truth_sims, expert_unigram_sims), pearsonr(expert_ground_truth_sims, expert_bigram_sims), pearsonr(expert_ground_truth_sims, expert_embedding_sims)

		novice_corrs = [novice_unigram_corr[0], novice_bigram_corr[0], novice_embedding_corr[0]]
		expert_corrs = [expert_unigram_corr[0], expert_bigram_corr[0], expert_embedding_corr[0]]
		for i, key in enumerate(['unigrams', 'bigrams', 'embeddings']):
			correlations['novice'][key].append(novice_corrs[i])
			correlations['expert'][key].append(expert_corrs[i])

		print("Ground truth correlation: novice unigrams %s, novice bigrams %s, novice embeddings %s" % (str(novice_unigram_corr), str(novice_bigram_corr), str(novice_embedding_corr)))
		print("Ground truth correlation: expert unigrams %s, expert_bigrams %s, expert embeddings %s" % (str(expert_unigram_corr), str(expert_bigram_corr), str(expert_embedding_corr)))
	for i, key in enumerate(['unigrams', 'bigrams', 'embeddings']):
		print(','.join([domain.title(), key, 'Novice,']), ','.join(map(str, correlations['novice'][key])))
		print(','.join([domain.title(), key, 'Expert,']), ','.join(map(str, correlations['expert'][key])))


if __name__ == "__main__":
	import sys
	import argparse

	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--checkpoints",nargs='+')
	parser.add_argument("--iters",nargs='+',type=int)
	parser.add_argument("--export_dir", type=str)
	parser.add_argument("--domain", type=str)

	args = parser.parse_args()

	checkpoints = args.checkpoints

	# Load all the metric_dicts for all of the checkpoints.
	metric_dicts = load_checkpoint_metrics(checkpoints, args.iters)

	# Unigram, bigram, and log production embedding entropies for solved tasks,
	# and for solved vs. unsolved tasks.
	if True:
		grammar_entropy_analysis(metric_dicts, args.export_dir)

	if False:
		# Updates the expert similarity matrices to calculate expected production uses under various grammars.
		updated_metric_dicts = update_checkpoint_metrics(checkpoints, max(args.iters), metric_dicts)
	# Unigram and bigram similarity analysis to ground truth.
	if False:
		grammar_distribution_similarity_analysis(updated_metric_dicts, args.export_dir, args.domain)


