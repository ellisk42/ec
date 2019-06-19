from dreamcoder.utilities import eprint
import random
import numpy as np

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
           Reshuffles across iterations - intended as benchmark comparison to test the task ordering."""
        def __init__(self, baseSeed=0): self.baseSeed = baseSeed

        def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
                if taskBatchSize is None:
                        taskBatchSize = len(tasks)
                elif taskBatchSize > len(tasks):
                        eprint("Task batch size is greater than total number of tasks, aborting.")
                        assert False
                
                # Reshuffles tasks in a fixed way across epochs for reproducibility.
                currEpoch = int(int(currIteration * taskBatchSize) / int(len(tasks)))

                shuffledTasks = tasks.copy() # Since shuffle works in place.
                random.Random(self.baseSeed + currEpoch).shuffle(shuffledTasks)

                shuffledTasksWrap = tasks.copy() # Since shuffle works in place.
                random.Random(self.baseSeed + currEpoch + 1).shuffle(shuffledTasksWrap)

                start = (taskBatchSize * currIteration) % len(shuffledTasks)
                end = start + taskBatchSize
                taskBatch = (shuffledTasks + shuffledTasksWrap)[start:end] # Wraparound nicely.

                return list(set(taskBatch))

class UnsolvedTaskBatcher:
        """At a given epoch, returns only batches of the tasks that have not been solved at least twice"""

        def __init__(self):
                self.timesSolved = {} # map from task to times that we have solved it
                self.start = 0

        def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
                assert taskBatchSize is None, "This batching strategy does not support batch sizes"

                for t,f in ec_result.allFrontiers.items():
                        if f.empty:
                                self.timesSolved[t] = max(0, self.timesSolved.get(t,0))
                        else:
                                self.timesSolved[t] = 1 + self.timesSolved.get(t, 0)
                return [t for t in tasks if self.timesSolved.get(t,0) < 2 ]
        
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

def kNearestNeighbors(ec_result, tasks, k, task):
        """Finds the k nearest neighbors in the recognition model logProduction space to a given task."""
        cosDistance = ec_result.recognitionModel.grammarLogProductionDistanceToTask(task, tasks)
        argSort = np.argsort(-cosDistance)# Want the greatest similarity.
        topK = argSort[:k]
        topKTasks = list(np.array(tasks)[topK])
        return topKTasks


class RandomkNNTaskBatcher:
        """Chooses a random task and finds the (taskBatchSize - 1) nearest neighbors using the recognition model logits."""
        def __init__(self):
                pass

        def getTaskBatch(self, ec_result, tasks, taskBatchSize, currIteration):
                if taskBatchSize is None:
                        taskBatchSize = len(tasks)
                elif taskBatchSize > len(tasks):
                        eprint("Task batch size is greater than total number of tasks, aborting.")
                        assert False

                if ec_result.recognitionModel is None:
                        eprint("No recognition model, falling back on random %d" % taskBatchSize)
                        return random.sample(tasks, taskBatchSize)
                else:
                        randomTask = random.choice(tasks)
                        kNN = kNearestNeighbors(ec_result, tasks, taskBatchSize - 1, randomTask)
                        return [randomTask] + kNN

class RandomLowEntropykNNTaskBatcher:
        """Choose a random task from the 10 unsolved with the lowest entropy, and finds the (taskBatchSize - 1) nearest neighbors using the recognition model logits."""
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
                        lowEntropyUnsolved = entropyRandomBatch(ec_result, unsolvedTasks, taskBatchSize, randomRatio=0)
                        randomTask = random.choice(lowEntropyUnsolved)
                        kNN = kNearestNeighbors(ec_result, tasks, taskBatchSize - 1, randomTask)
                        return [randomTask] + kNN


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





                

