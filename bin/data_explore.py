try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import sys,os
from dreamcoder.type import Context, arrow, tint, tlist, tbool, UnificationFailure
import argparse
from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloader


import mlb


parser = argparse.ArgumentParser()

parser.add_argument('T',
                    choices = [1,2,3],
                    type=int,
                    help='which dataset to use')
parser.add_argument('lambda_depth',
                    type=int,
                    help='max depth of lambdas')
parser.add_argument('--original',
                    action='store_true',
                    help='use this if you want to see the original deepcoder data without mutated lambdas')

args = parser.parse_args()

print(f"T: {args.T}")
print(f"lambda_depth: {args.lambda_depth}")
print(f"original: {args.original}")

assert args.lambda_depth > 1, "there are no valid depth 1 lambdas because we exclude identity function + constants"

#NUM_TASKS = 20
NUM_MUTATIONS = 10

allowed_requests = [arrow(tlist(tint),tlist(tint))]
taskloader = DeepcoderTaskloader(
    f'dreamcoder/domains/list/DeepCoder_data/T{args.T}_A2_V512_L10_train_perm.txt',
    allowed_requests=allowed_requests,
    repeat=False, # means the data will not loop when it runs out
    num_tasks=None, # no limit on number of tasks we can take
    expressive_lambdas=(not args.original),
    lambda_depth=args.lambda_depth,
    num_mutated_tasks=NUM_MUTATIONS,
    )

tasks = taskloader.getTasks(100, ignore_eof=True)