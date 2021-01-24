from time import time
with open('dreamcoder/domains/list/DeepCoder_data/T3_A2_V512_L10_train_perm.txt') as f:
    start = time()
    i=0
    while (l:= f.readline()) != '':
        i += 1
        if l.strip() == '':
            print("fuck")
    print(f"done at rate {i/(time()-start):.5f}s with {i} lines")
print("done")

