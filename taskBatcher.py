from utilities import eprint
import random
import numpy as np

def defaultBatching(tasks, taskBatchSize, currIteration):
	"""Returns the currIteration taskBatch of size taskBatchSize without reordering."""
	start = (taskBatchSize * currIteration) % len(tasks)
	end = start + taskBatchSize
	taskBatch = (tasks + tasks)[start:end] # Handle wraparound.
	return taskBatch

class DefaultTaskBatcher:
	"""Iterates through task batches of the specified size. Defaults to all tasks if taskBatchSize is None."""

	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		if taskBatchSize is None:
			taskBatchSize = len(tasks)
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False

		return defaultBatching(tasks, taskBatchSize, currIteration), None

class RandomTaskTaskBatcher:
	"""Returns a randomly sampled task batch of the specified size. Defaults to all tasks if taskBatchSize is None."""

	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		if taskBatchSize is None:
			taskBatchSize = len(tasks)
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False

		return random.sample(tasks, taskBatchSize), None

class RecognitionDensityTaskBatcher:
	"""Reranks tasks according to recognition model for density, then returns the top-k tasks as a batch.
	   Only works for context-free grammars. """

	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		if taskBatchSize is None:
			taskBatchSize = len(tasks)
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False

		# Use the recognition model from the previous result if we have it
		if ec_result.recognitionModel is None:
			eprint("No recognition model, falling back on default ranking.")
			return defaultBatching(tasks, taskBatchSize, currIteration), None
		else:
			eprint("Recognition model found, reranking tasks by probability density.")
			
			# Rank based on least entropy if applicable and choose the top k.
			entropies = ec_result.recognitionModel.taskGrammarEntropies(tasks)
			entropiesArray = np.array([entropies[task] if entropies[task] is not None else np.nan for task in tasks ])
			taskRanks = np.argsort(entropiesArray)

			eprint(taskRanks.shape, np.array(tasks).shape, np.array(tasks)[taskRanks].shape)
			rerankedTasks = list(np.array(tasks)[taskRanks])
			taskEntropies = list(np.array(entropiesArray)[taskRanks])
		return rerankedTasks[:taskBatchSize], taskEntropies[:taskBatchSize]


		