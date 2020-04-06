import pickle

fname = "experimentOutputs/draw/2019-11-10T22:47:09.031267/parsesflat_S12_13_test_2.pickle"
with open(fname, "rb") as f:
	parses = pickle.load(f)

import random
parses = random.sample(parses, 10000)

fnamesave = fname + "2"
with open(fnamesave, "wb") as f:
	pickle.dump(parses, f)
