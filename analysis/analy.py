## various analyses to compare dreamcoder and behavior

import pyxdameraulevenshtein as dl

def stringDist(a, b):
    # rerutns value between 0 and 1 (1 is max difference)
    # first map all the items to idx identifiers (e.g. from  ["C1", "C2", "L1", "L2"] to [1,2,3,4]
#     alphabet = ["C1", "C2", "L1", "L2"]
#     A = [alphabet.index(aa) for aa in a]
#     B = [alphabet.index(bb) for bb in b]
    # e.g,, dl.damerau_levenshtein_distance(["C2", "C0", "L1", "L2"], ["C9", "LLL", "C", "L"])

    return dl.normalized_damerau_levenshtein_distance(a,b)


