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

class RandomShuffleTaskBatcher:
	"""Randomly shuffles the task batch first, and then iterates through task batches of the specified size like DefaultTaskBatcher.
	   Uses a fixed shuffling across iterations - intended as benchmark comparison to test the task ordering."""
	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		if taskBatchSize is None:
			taskBatchSize = len(tasks)
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False
		
		# Shuffles tasks with a set seed across iterations.
		shuffledTasks = tasks.copy() # Since shuffle works in place.
		random.Random(0).shuffle(shuffledTasks)

		start = (taskBatchSize * currIteration) % len(shuffledTasks)
		end = start + taskBatchSize
		taskBatch = (tasks + tasks)[start:end] # Handle wraparound.
		return taskBatch


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

def entropyRandomBatch(ec_result, tasks, taskBatchSize, randomRatio):
	numRandom = int(randomRatio * taskBatchSize)
	numEntropy = taskBatchSize - numRandom

	eprint("Selecting top %d tasks from the %d overall tasks given lowest entropy." % (taskBatchSize, len(tasks)))
	eprint("Will be selecting %d by lowest entropy and %d randomly." %(numEntropy, numRandom))
	taskGrammarEntropies = ec_result.recognitionModel.taskGrammarEntropies(tasks)
	sortedEntropies = sorted(taskGrammarEntropies.items(), key=lambda x:x[1])

	entropyBatch = [task for (task, entropy) in sortedEntropies[:numEntropy]]
	randomBatch = random.sample([task for (task, entropy) in sortedEntropies[numEntropy:]], numRandom)
	batch = entropyBatch + randomBatch

	return batch


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

		if ec_result.recognitionModel is None:
			eprint("No recognition model, falling back on random %d tasks from the remaining %d" %(taskBatchSize, len(unsolvedTasks)))
			return random.sample(unsolvedTasks, taskBatchSize)
		else:
			return entropyRandomBatch(ec_result, unsolvedTasks, taskBatchSize, randomRatio=0)

class UnsolvedRandomEntropyTaskBatcher:
	"""Returns tasks that have never been solved at any previous iteration.
	   Given a task batch size, returns a mix of unsolved tasks with percentRandom 
	   selected randomly and the remaining selected by lowest entropy.""" 
	def __init__(self):
		pass

	def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
		unsolvedTasks = [t for t in tasks if ec_result.allFrontiers[t].empty]

		if taskBatchSize is None:
			return unsolvedTasks
		elif taskBatchSize > len(tasks):
			eprint("Task batch size is greater than total number of tasks, aborting.")
			assert False

		if ec_result.recognitionModel is None:
			eprint("No recognition model, falling back on random %d tasks from the remaining %d" %(taskBatchSize, len(unsolvedTasks)))
			return random.sample(unsolvedTasks, taskBatchSize)
		else:
			return entropyRandomBatch(ec_result, unsolvedTasks, taskBatchSize, randomRatio=.5)

			

		

