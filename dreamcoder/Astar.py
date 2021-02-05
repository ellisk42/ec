from dreamcoder.pnode import FoundSolution, PNode
import sys
import time
from dataclasses import dataclass, field
from dreamcoder.matt.plot import SearchTry
from dreamcoder.matt.util import *
@dataclass(order=True)
class HeapItem:
    cost: float
    pcost: float = field(compare=False)
    hole: PNode = field(compare=False)
    prod = field(compare=False)
    verify_str: str = field(compare=False)

def astar_search(root, phead, vhead, timeout, max_depth):
    """
    Feel free to pass in a tree that is already partially filled in
    """
    assert root.ntype.output
    # self.critic_coeff = solver_cfg.critic_coeff
    # self.max_depth = solver_cfg.max_depth

    sys.setrecursionlimit(5000)

    q = Heap(max_size=1200000,reset_to_size=1000000)
    seen = set()
    nodes_expanded = 0
    tstart = time.time()

    next = root.root()
    prev_pcost = 0

    while True:
        if time.time() - tstart > timeout:
            return SearchTry(time=timeout, nodes_expanded=nodes_expanded, soln=None)

        verify_str = next.root_str()

        # enumerate cand actions
        hole, prods, pcosts = phead.enumerate_actions(next, max_depth)

        # value costs on that whole batch of actions
        try:
            vcosts = vhead.values(hole,prods)
        except FoundSolution as fs:
            soln = fs.p
            return SearchTry(time=time.time()-tstart, nodes_expanded=nodes_expanded, soln=soln)


        assert verify_str == next.root_str(), "vhead.values() seems to have modified the hole"
        assert len(vcosts) == len(prods) == len(pcosts)

        hashed_hole = hole.marked_str()

        for prod, pcost, vcost in zip(prods,pcosts,vcosts):
            pcost += prev_pcost # pcost accumulates but vcost doesnt
            cost = pcost - critic_coeff * vcost
            q.push(HeapItem(
                cost=cost,
                pcost=pcost,
                hole=hole,
                prod=prod,
                verify_str=verify_str,
            ))
            nodes_expanded += 1
            
            if (hashed_hole,prod) in seen:
                assert False, "im doubtful that this would even ever fire??"
                continue
            seen.add((hashed_hole,prod))

        # pop a hole off the heap, clone it, fill it in w the production from the heap (this is lazy for efficiency)
        # and we done.
        heap_item = q.pop_min()
        hole = heap_item.hole
        assert hole.ntype.hole
        prod = heap_item.prod
        prev_pcost = heap_item.pcost
        assert heap_item.verify_str == hole.root_str(), "the hole seems to have been modified while in the heapitem"
        next = hole.clone() # duplicates by cloning every pnode to make a fully independent tree, 
        next.expand_to(prod) # this is our new guy!
        next = next.root()
