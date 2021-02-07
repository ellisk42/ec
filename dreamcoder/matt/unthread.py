def unthread():
    """
    Disables cpu numpy/torch thread parallelism
    """
    import os,sys
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

    if 'numpy' in sys.modules:
        import mlb # this import cant come earlier in case `mlb` imports numpy
        mlb.yellow("warning: unthread() might not work properly if done after importing numpy")
    else:
        import mlb # this import cant come earlier in case `mlb` imports numpy
        mlb.green('successfullly limited numpy and torch threads to 1')

    import torch
    torch.set_num_threads(1)

def set_deterministic(seed):
    # imports must happen in here so unthread() can be called before doing any imports
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    # warning: these may slow down your model
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    import mlb
    mlb.green('Set numpy, torch, and random to be deterministic')