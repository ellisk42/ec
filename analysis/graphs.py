## plots about dreamcoder training
import matplotlib.pyplot as plt

def plotNumSolved(result):
	fig = plt.figure()
	plt.xlabel("wake step")
	plt.ylabel('tasks solved')
	plt.plot(result.hitsAtEachWake, '-ok')
	return fig

def plotAllTasks(DAT, trainortest="train"):
    import dreamcoder.domains.draw.primitives as P
    import math
    NCOL = 6
    print("NOTE: whether is solved is by checking the last iteration")

    if trainortest=="train":
        tasks = DAT["tasks"]
        tasknames = DAT["programnames"]
    elif trainortest=="test":
        tasks = DAT["testtasks"]
        tasknames = DAT["programnames_test"]

    N = len(tasks)
    ncol = NCOL
    nrow = math.ceil(N/ncol)

    fig = plt.figure(figsize=(ncol*3, nrow*3))

    for i, (t, name) in enumerate(zip(tasks, tasknames)):
        # 1) Plot this ground truth program
        ax = plt.subplot(nrow, ncol, i+1)
        P.plotOnAxes(t.strokes, ax)

        solved = DAT["taskresultdict"][name]
            # 
        # if len(DAT["result"].frontiersOverTime[t][-1])>0:
        #     solved=True
        # else:
        #     solved=False

        if solved:
            c = "b"
        else:
            c = "r"

        plt.title("{}, {}, s={}".format(i, name, solved), color=c)

    return fig
