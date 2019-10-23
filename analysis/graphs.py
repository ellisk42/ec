## plots about dreamcoder training
import matplotlib.pyplot as plt

def plotNumSolved(result):
	plt.figure()
	plt.xlabel("wake step")
	plt.ylabel('tasks solved')
	plt.plot(result.hitsAtEachWake, '-ok')



def 