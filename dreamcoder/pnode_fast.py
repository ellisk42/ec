

from dreamcoder.pnode import *
from dreamcoder.matt.util import *

import functools


def main(node):
  node = label_ctxs(node)
  batched_ctx_encode(node)
  toplevel_subst(node) # toplevel vars -> prims however same contexts as before since lambdas could capture from it
  fold_concrete(node)
  batcher = Batcher(node)
  batcher.saturate()
  

def fold_concrete(p:PNode):
  """
  folds all concrete subtrees including variables into PRIM nodes with the given value
  """

def label_ctxs(p:PNode):
  """
  label every node with a .ctx Context and use batched_ctx_encode() at the end to precompute them all in a batch
  And yes this ctx shd have placeholders for anything in a lambda
  """

def batched_ctx_encode():
  """
  takes a list of Context objects, removes identical ones (that share memory) and computes all thier .encode()s at once
  """

def toplevel_subst():
  """
  Runs toplevel subst so `OUT -> ABS -> MAP ...` becomes `OUT -> MAP ...` and any instances of `$i` get replaced with their actual values as PRIM.
  """

def up_edge(edge):
  return edge[1] < edge[0]

def get_nodes_edges(node):
  """
  if higher in tree, you should have a lower number.
  So this is a BFS of the tree
  """
  id=0
  nodes = []
  edges = []
  worklist = [node]
  while len(worklist) > 0:
    nodes += worklist
    for node in worklist:
      node.id = id
      if not node.is_root():
        edges.append((node.id,node.parent.id))
      id += 1
    # move worklist to be one step deeper in tree
    worklist = flatten(node.children() for node in worklist)
  
  edges += [(dst,src) for src,dst in edges] # add all reverse edges

  return nodes,edges



class Batcher:
  def __init__(self, root:PNode):
    self.nodes, self.edges = get_nodes_edges(root)
    self.edges_todo = self.edges[:] # :: [(src,dst)]
    self.edges_done = [] # :: [(src,dst)]
    self.edges_done_results = [] # :: [Tensor] and is zippable with edges_done

    self.neighbors = functools.cache(self.neighbors) # do this instead of decorator so its a per-instance thing and also gets garbage collected
    self.necessary_edges = functools.cache(self.necessary_edges)
  
  def saturate(self):
    
    while len(self.edges_todo) > 0:
      worklist = []
      for src,dst in self.edges_todo:
        if all(edge in self.edges_done for edge in self.necessary_edges(src)):
          worklist.append((src,dst))
      assert len(worklist) > 0

      worklist,worklist_fromcache = self.split_by_cache(worklist)
      self.compute_edges(worklist)
      rest = self.try_cache(worklist_fromcache)
      assert len(rest) == 0

  def try_cache(self,edges):
    """
    returns whatever edges it cant find in the cache, .finish()es the others directly
    """

    # self.finish_edge(edge,vec)
    
  def finish_edge(self,edge,vec):
      assert vec.dim() == 2
      self.edges_todo.remove(edge)
      self.edges_done.append(edge)
      self.edges_done_results.append(vec)
  
  def finish_edges(self,edges,vecs):
    """
    vecs can be 3d tensor or list of 2d tensors, doesnt matter since we iterate
    over the outer dimension.
    """
    for edge,vec in zip(edges,vecs):
      self.finish_edge(edge,vec)
      
  def split_by_cache(self,edges):
    """
    return (edges_to_compute,edges_to_try_cache)
    these two lists together form the original `edges` list, but the key is if you
    were to compute all the edges in edges_to_compute you could then use your cache
    to retrieve everything in edges_to_try_cache

    Note that this takes into account boht the current cache + deduping the `edges` list
    """

  def compute_edges(self,edges):
    """
    compute the tensors on all these edges bc they're ready to be computed, and .finish() them via compute_edges_ntype()
    """
    by_ntype = group_by(edges, lambda e: self.nodes[e[0]].ntype)
    
    for ntype, edges in by_ntype.items():
      self.compute_edges_ntype(ntype, edges)
  
  def compute_edges_ntype(self, ntype, edges):
    """
    assumes src node in all edges have the same ntype, computes and .finish()es them
    """
    {
      NType.PRIM : self.compute_edges_prim,
      NType.VAR : self.compute_edges_var,
      NType.HOLE : self.compute_edges_hole,
      NType.OUTPUT : self.compute_edges_output,
      NType.ABS : self.compute_edges_abs,
      NType.APP : self.compute_edges_app,
    }[ntype](edges)
      
  def compute_edges_prim(self, edges):
    """
    PRIM
    extract the values, duplicate each one num_exs times, flatten this out,
    run it through the encoder (tricking it into thinking num_exs is batch_size*num_exs)
    then reshape to get the batch dimension back out

    sadly our current encodeValue can only batch together values of the same type since it takes
    different paths for list vs nonlist.
    """
    by_type = group_by(edges, lambda e: type(self.nodes[e[0]].value))
    for ty,edges in by_type.items():
      values = [self.nodes[src].value for (src,dst) in edges]
      values = flatten([[v]*sing.num_exs for v in values])
      res = sing.model.abstraction_fn.encoder.encodeValue(values)
      res = rearrange(res,'(batch exs) H -> batch exs H',batch=len(edges),exs=sing.num_exs)
      self.finish_edges(edges,res)

  def compute_edges_var(self, edges):
    """
    VAR

    Note this is only for variables that we've chosen to encode as placeholders and not
    run subst on.
    """
    by_i = group_by(edges, lambda e: self.nodes[e[0]].i)
    for i,edges in by_i.items():
      #TODO sometime migrate this and beval() to just be `index_nm(i)` used universally
      res = sing.model.abstract_transformers.lambda_index_nms[i]().expand(len(edges),sing.num_exs,-1)
      self.finish_edges(edges,res)

  def compute_edges_hole(self, edges):
    """
    HOLE
    batched_ctx_encode() and label_ctxs() helpfully precomputed our .ctx objects with encode() precomputation included
    """
    by_tp = group_by(edges, lambda e: self.nodes[e[0]].tp.show(True))
    for tp,edges in by_tp.items():
      ctxs = torch.stack([self.nodes[edge[0]].ctx.encode() for edge in edges]) # the .encode() is already cached btw
      res = sing.model.abstract_transformers.hole_nms[tp](ctxs)
      self.finish_edges(edges,res)

  def compute_edges_output(self, edges):
    """
    OUT
    """
    res = [self.nodes[edge[0]].task.output_features() for edge in edges]
    self.finish_edges(edges,res)

  def compute_edges_abs(self, edges):
    """
    ABS
    """
    # grab the one incoming edge that we already know
    input_vecs = []
    for e in edges:
      input_edge = self.necessary_edges(*e)[0] # grab the one necessary edge
      input_vecs.append(self.get_done_vec(*input_edge))

    by_dir = group_by(zip(edges,input_vecs), lambda edge_invec: up_edge(edge_invec[0]))

    ### UP
    edges, input_vecs = zip(*by_dir[True]) # [(edge,invec)] -> (edges,invecs)
    #TODO stack input_vecs and run thru a lam_up() NM for prettier semantics
    res = input_vecs
    self.finish_edges(edges,res)

    ### DOWN
    edges, input_vecs = zip(*by_dir[False]) # [(edge,invec)] -> (edges,invecs)
    #TODO add a lam_down() NM for prettier semantics
    res = input_vecs
    self.finish_edges(edges,res)

  def compute_edges_app(self, edges):
    """
    APP
    """
    batch_known = []
    batch_known_labels = []
    batch_target = []
    for (src,dst) in edges:
      neis = sorted(self.neighbors(src)) # take advantage that out < f < arg0 < arg1 < arg2 < arg3
      neis = list() # now this is [(nei_id,label)]
      labels = []
      known = []
      target = None
      for (nei,label) in zip(neis,['out','fn',0,1,2,3]):
        if nei == dst:
          target = label
          continue
        labels.append(label)
        known.append(self.get_done_vec(nei,src))
      batch_known.append(known)
      batch_known_labels.append(labels)
      batch_target.append(target)

    res = sing.model.apply_nn.batch_forward(batch_known, batch_known_labels, batch_target)
    self.finish_edges(edges,res)      

    
  def get_done_vec(self,src,dst):
    return self.edges_done_results[self.edges_done.index((src,dst))]

  def neighbors(self,id):
    return {dst for (src,dst) in self.edges if src==id}

  def necessary_edges(self,src,dst):
    return {(nei,src) for nei in  self.neighbors(src) - {dst}}




