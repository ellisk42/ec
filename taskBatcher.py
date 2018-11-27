from utilities import eprint
import random

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
		

		start = (taskBatchSize * currIteration) % len(tasks)
		end = start + taskBatchSize
		taskBatch = (tasks + tasks)[start:end] # Handle wraparound.
		return taskBatch

class RandomTaskBatcher:
	"""Returns a randomly sampled task batch of the specified size. Defaults to all tasks if taskBatchSize is None."""

	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		if taskBatchSize is None:
			taskBatchSize = len(tasks)
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False

		return random.sample(tasks, taskBatchSize)

class UnsolvedTaskBatcher:
	"""Returns tasks that have never been solved at any previous iteration. If a batch size is passed in, returns
	   a randomly sampled task batch of the specified size."""

	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		unsolvedTasks = [t for t in tasks if ec_result.allFrontiers[t].empty]

		if taskBatchSize is None:
			return unsolvedTasks
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False

		eprint("Randomly sampling %d tasks from the unsolved %d remaining tasks." % (taskBatchSize, len(unsolvedTasks)))
		return random.sample(unsolvedTasks, taskBatchSize)

class UnsolvedEntropyTaskBatcher:
	"""Returns tasks that have never been solved at any previous iteration.
	   Given a task batch size, returns the unsolved tasks with the lowest entropy."""
	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		unsolvedTasks = [t for t in tasks if ec_result.allFrontiers[t].empty]

		if taskBatchSize is None:
			return unsolvedTasks
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False

		# NOT YET DONE.
		eprint("Selecting top %d tasks from the unsolved %d remaining tasks given lowest entropy." % (taskBatchSize, len(unsolvedTasks)))
		return random.sample(unsolvedTasks, taskBatchSize)

