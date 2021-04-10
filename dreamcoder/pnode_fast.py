

from dreamcoder.pnode import *
from dreamcoder.matt.util import *

import functools


def fast(roots):
  for node in roots:
    rewrite(node)
    label_ctxs(node)
    batched_ctx_encode([n.ctx for n in node.all_nodes()])
    fold_concrete(node)
  batcher = Batcher(roots)
  batcher.saturate()
  return batcher.nodes, batcher.edges, batcher.edges_done_results



def fold_concrete(p:FPNode):
  """
  folds all concrete subtrees including variables into EXWISE nodes
  """

def label_ctxs(p:FPNode):
  """
  label every node with a .ctx EWContext thats all concrete or placeholders
  """
  

def label_concrete(p:FPNode):
  """
  label every node with a .ctx EWContext thats all concrete or placeholders
  """


def batched_ctx_encode(ctxs):
  """
  takes a list of Context objects, removes identical ones (that share memory) and computes all thier .encode()s at once
  """
  # TODO make this actually parallel some time!
  ctxs = list(set(ctxs))
  for ctx in ctxs:
    ctx.encode()


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
  def __init__(self, roots):
    self.roots = roots
    self.nodes = []
    self.edges = []
    for root in roots:
     nodes,edges = get_nodes_edges(root)
     self.nodes += nodes
     self.edges += edges
    
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







class FPNode:
    def __init__(self, ntype, tp, parent, ctx_tps):
        super().__init__()
        if isinstance(parent,PTask):
            self.task = parent
            self.parent = self # root node. We dont use None as the parent bc we want towards=None to be reserved for our Cache emptiness
        else:
            self.task = parent.task
            self.parent = parent

        self.ntype = ntype
        self.tp = tp
        self.ctx_tps = ctx_tps

    def is_root(self):
        return self.parent == self
    def all_nodes(self):
        return [self] + self.children(recursive=True)
    def root_str(self):
        """
        Useful for when you wanna ensure that the whole program hasnt been inplace
        modified somehow over some period
        """
        return str(self.root())
    def subtree_str(self):
        """
        Useful for when you wanna ensure that the subtree below this hasnt been modified
        over some period
        """
        return str(self)
    def marked_str(self):
        """
        A unique little string that shows the whole program but with `self` marked clearly with double brackets [[]]
        """
        return self.root().marked_repr(self)
    def expand_to(self, prim):
        """
        prim :: Primitive(Program) | Index(Program) | Examplewise

        Call this on a hole to transform it inplace into something else.
         - prim :: Primitive | Index
         - If prim.tp.isArrow() this becomes an APP with the proper holes added for cildren
         - You can never specify "make an Abstraction" because those get implicitly created by build_hole()
            whenever a hole is an arrow type. Note that a hole being an arrow type is different from `prim`
            being an arrow type. An arrow prim can fill a hole that has its return type, however if the arrow
            prim futhermore has arguments that are arrows, those are where the abstractions will be created (HOFs).
         - if `self` is an abstraction it'll get unwrapped to reveal the underlying hole as a convenience (simplifies
            a lot of use cases and recursion cases).

        """
        self = self.unwrap_abstractions()
        assert self.ntype.hole
        assert not self.tp.isArrow(), "We should never have an arrow for a hole bc it'll instantly get replaced by abstractions with an inner hole"
        if isinstance(prim,Examplewise):
          self.ew = prim
          self.ntype = NType.EXWISE
        elif prim.isIndex:
            assert self.ctx_tps[prim.i] == self.tp # the ctx tp is the same as our tp
            self.i = prim.i
            self.ntype = NType.VAR
        elif prim.isPrimitive:
            assert prim.tp.returns() == self.tp # in order to fill the hole in a valid way
            if not prim.tp.isArrow():
                # PRIM case
                self.ntype = NType.PRIM
                self.prim = prim
                self.name = prim.name
                self.value = prim.value
            else:
                # APP case
                self.ntype = NType.APP
                # make self.fn as a PRIM 
                self.fn = FPNode(NType.PRIM, tp=prim.tp, parent=self, ctx_tps=self.ctx_tps)
                self.fn.prim = prim
                self.fn.name = prim.name
                self.fn.value = UncurriedFn(prim.value,name=prim.name)
                # make holes for args
                self.xs = [self.build_hole(arg_tp) for arg_tp in prim.tp.functionArguments()]
        else:
            raise TypeError
        
    def build_hole(self, tp):
        """
        Make a new hole with `self` as parent (and `ctx` calculated from `self`)
        This also handles expanding into Abstractions if tp.isArrow()
        """
        if not tp.isArrow():
            return FPNode(NType.HOLE, tp, parent=self, ctx_tps=self.ctx_tps)
        
        arg_tp = tp.arguments[0] # the input arg to this arrow
        res_tp = tp.arguments[1] # the return arg (which may be an arrow)

        abs = FPNode(NType.ABS, tp, parent=self, ctx_tps=(arg_tp,*self.ctx_tps))
        inner_hole = abs.build_hole(res_tp)
        abs.body = inner_hole
        abs.argc = 1  # TODO can change

        return abs # and return our abstraction

    def into_hole(self):
        """
        reverts a FPNode thats not a hole back into a hole. If you still want to keep around
        a ground truth non-hole version of the node you probably want FPNode.hide() instead.
        """
        assert not self.ntype.hole
        assert not self.ntype.abs, "you never want an arrow shaped hole buddy"
        assert not self.ntype.output, "wat r u doin"
        for attr in ('prim','name','value','fn','xs','i','body','tree','ew'):
            if hasattr(self,attr):
                delattr(self,attr) # not super important but eh why not
        self.ntype = NType.HOLE

    def expand_from(self, other, recursive=True):
        """
        like expand_to but taking another FPNode instead of a Prod.
        recursive=True means itll recurse so you can fill in whole subtrees.
        cache wont be copied, you should call self.copy_cache_from(other, recursive=True) at the
            end if you want that
        Abstractions and Outputs get unwrapped
        """
        if self.ntype.output:
            self = self.tree
        if other.ntype.output:
            other = other.tree
        self = self.unwrap_abstractions()
        other = other.unwrap_abstractions()
        assert self.ntype.hole
        assert self.tp == other.tp
        assert not other.ntype.output, "holes never expand into outputs"
        assert not other.ntype.abs, "this should have been unwrapped"

        if other.ntype.hole:
            return # we're already a hole! Nothing to change
        
        prod = other.get_prod()
        self.expand_to(prod)
        assert self.ntype == other.ntype, "expansion didnt yield the expected ntype"
        
        if recursive and self.ntype.app:
            # APP is the only case w children but we dont do the `fn` since its just a prefilled prim
            for c,o in zip(self.xs,other.xs):
                c.expand_from(o,recursive=True)


        
    def check_solve(self):
        """
        check if we're a solution to the task
        """
        if self.root().has_holes:
            return False
        try:
            res = self.root().beval(None).concrete
            assert res is not None
            return res == self.task.outputs.concrete
        except InvalidSketchError:
            return False

    @staticmethod
    def from_ptask(ptask: PTask):
        """
        Create a tree shaped like:
                FPNode(OUTPUT)
                     |
                 FPNode(APP)
                   /    \                  
           FPNode(ABS)    FPNode(EXWISE: input[0]) 
                |
                |
           FPNode(HOLE)
        
        Or however many abstractions make sense for the given ptask
        Returns the root of the tree (the output node)
        """
        root = FPNode(NType.OUTPUT, tp=ptask.request.returns(), parent=ptask, ctx_tps=())
        app = FPNode(NType.APP, tp=ptask.request.returns(), parent=root, ctx_tps=())
        root.tree = app
        app.fn = app.build_hole(ptask.request) # build an ABS
        app.xs = []
        for input_ew,tp in zip(ptask.inputs, ptask.request.functionArguments()):
            arg = FPNode(NType.EXWISE, tp=tp, parent=app, ctx_tps=())
            arg.ew = input_ew
            app.xs.append(arg)
        return root
    
    @staticmethod
    def from_dreamcoder(p: Program, task:Task):
        """
        Given a dreamcoder Program and Task, make an equivalent FPNode
        and associated PTask. Returns the root (an output node).
        """
        # create a full program from the top level task and program
        # meaning this will be our ntype.output node
        root = FPNode.from_ptask(PTask(task))
        root.tree.abs.expand_from_dreamcoder(p)
        assert str(root.tree) == str(p), "these must be the same for the sake of the RNN which does str() of the pnode"
        return root

    def expand_from_dreamcoder(self, p: Program):
        """
        Like expand_to() except for replacing the hole with a subtree equivalent
        to the given dreamcoder program. All abstractions in both `self` and `p`
        will be unwrapped to get at the underlying holes so it's okay to pass in
        abstractions (this simplifies recursion cases).
        """
        self = self.unwrap_abstractions()
        assert self.ntype.hole

        # unwrap abstractions
        while p.isAbstraction:
            p = p.body

        if p.isPrimitive or p.isIndex:
            # p is a Primitive or Index
            self.expand_to(p)
        elif p.isHole:
            pass # we already are a hole!
        elif p.isAbstraction:
            assert False # can't happen bc of the `while p.isAbstraction` unwrapping above
        elif p.isApplication:
            # application. We expand each hole we create (the f hole and the xs holes)
            f, xs = p.applicationParse()
            assert f.isPrimitive
            self.expand_to(f) # expand to an APP with proper fn and holes for args
            for x,x_ in zip(self.xs,xs):
                x.expand_from_dreamcoder(x_)
        else:
            raise TypeError
    
    def __str__(self):
        if self.ntype.abs:
            return f'(lambda {self.body})'
        elif self.ntype.app:
            args = ' '.join(str(arg) for arg in self.xs)
            return f'({self.fn} {args})'
        elif self.ntype.prim:
            return f'{self.name}'
        elif self.ntype.var:
            return f'${self.i}'
        elif self.ntype.hole:
            return f'<HOLE>'
        elif self.ntype.output:
            return f'{self.tree}'
        elif self.ntype.exwise:
            if hasattr(self,'history'):
              return 'EW{'+self.history+'}'
            else:
              return 'EW'
        else:
            raise TypeError
    def __repr__(self):
        return f'{self.ntype.name}({self.tp}): {self.marked_str()}'
    def marked_repr(self,marked_node):
        """
        A recursive repr() function like repr() but if it encounters 'marked_node' that node
        will be printed with [[]] brackets around it
        """
        if self.ntype.abs:
            res = f'(lambda {self.body.marked_repr(marked_node)})'
        elif self.ntype.app:
            args = ' '.join(arg.marked_repr(marked_node) for arg in self.xs)
            res = f'({self.fn.marked_repr(marked_node)} {args})'
        elif self.ntype.prim or self.ntype.var or self.ntype.hole or self.ntype.exwise:
            res = str(self)
        elif self.ntype.output:
            res = self.tree.marked_repr(marked_node)
        else:
            raise TypeError
        if self is marked_node:
            return f'[[{res}]]'
        return res
    @property
    def in_HOF_lambda(self): # TODO worth changing, p specific to this dsl
        return self.get_zipper().count('body') > len(self.task.inputs)

    def beval_single_concrete(self, ctx):
        """
        non-exwise beval. If you call it on an output node or an abs node youll
        get a python lambda you can feed args into.
        """

        sing.scratch.beval_print(f'beval_single {self} with ctx={ctx}', indent=True)

        def printed(res):
            assert res is not None
            sing.scratch.beval_print(f'{mlb.mk_green("->")} {res}', dedent=True)
            return res


        assert not self.has_holes


        if self.ntype.output:
            assert ctx is None
            return printed(self.tree.beval_single_concrete(SinglesContext())) # this will just convert the toplevel ABS into a closure
        
        assert ctx is not None

        if self.ntype.prim:
            """
            for both fn prims and other prims
            """
            return printed(self.value)

        elif self.ntype.var:
            return printed(ctx[self.i])

        elif self.ntype.hole:
            assert False

        elif self.ntype.abs:
            return printed(SinglesClosure(abs=self,enclosed_ctx=ctx))

        elif self.ntype.app:
            fn = self.fn.beval_single_concrete(ctx)
            args = [arg.beval_single_concrete(ctx) for arg in self.xs]
            return printed(fn(*args))
    
    def beval(self, ctx):
        """
        call like root.beval(ctx=None) and the output node will fill the right ctx for you.
        """
        sing.scratch.beval_print(f'{mlb.mk_blue("beval")}({self.ntype}) {self} with ctx={ctx}', indent=True)

        if hasattr(self,'needs_ctx'):
            self.needs_ctx = ctx
            sing.scratch.beval_print('[hit beval needs ctx]')

        def printed(res):
            assert res is not None
            sing.scratch.beval_print(f'{mlb.mk_green("->")} {short_repr(res)}', dedent=True)
            return res
                
        def no_cache():
            if self.ntype.output:
                """
                beval on the output sets up the ctx properly and skips over the abstractions into their bodies and
                executes the bodies in the proper context.
                """
                assert ctx is None
                body,i = self.tree.unwrap_abstractions(count=True)
                assert len(self.task.ctx) == i
                return body.beval(self.task.ctx)
            
            assert ctx is not None

            if self.ntype.prim:
                """
                even for fn primitives, just always return an EW(concrete=...)
                """
                return Examplewise(concrete=[self.value for _ in range(sing.num_exs)])

            elif self.ntype.var:
                ew = ctx[self.i]
                if not ew.placeholder:
                    return ew # normal case
                else:
                    # encode a free var
                    return Examplewise(abstract=ew.encode_placeholder(self.i))


            elif self.ntype.hole:
                return Examplewise(abstract=sing.model.abstract_transformers.hole_nms[self.tp.show(True)](ctx.encode()))

            elif self.ntype.abs:
                return Examplewise(closure=EWClosure(abs=self,enclosed_ctx=ctx))

            elif self.ntype.app:
                fn = self.fn.beval(ctx)
                args = [arg.beval(ctx) for arg in self.xs]
                

                if (ctx.no_free and # no free vars
                    fn.can_be_concrete and # fn can be concrete
                    all(x.can_be_concrete for x in args) # args can be concrete
                    ):
                    """
                    Concrete application!
                    This uses beval_single_concrete implicitly since it
                        does ew.split() on all args as well as the fn which
                        means even if the fn was a closure it became a list of
                        SinglesClosures with SinglesContexts (so even our current
                        context was converted even tho we dont explicitly convert
                        it here).
                    """
                    sing.scratch.beval_print(f'[concrete apply]')

                    singles_fns = fn.split()
                    singles_args = list(zip(*[ew.split() for ew in args])) # [argc,num_exs] -> [num_exs,argc]
                    assert len(singles_fns) == len(singles_args) == sing.num_exs
                    res = [fn(*args) for fn,args in zip(singles_fns,singles_args)]
                    return Examplewise(concrete=res)



                """
                Abstract call!
                Abstract the fn and args (they were already beval()'d) then label them and pass to apply_nn
                """
                ### * V1 * ###
                sing.scratch.beval_print(f'[abstract apply]')
                sing.stats.fn_called_abstractly += 1
                assert self.fn.ntype.prim, "would work even if this wasnt a prim, but for v1 im leaving this here as a warning"
                
                fn_embed = fn.get_abstract() # gets the Parameter vec for that primitive fn
                args_embed = [arg.get_abstract() for arg in args]
                labelled_args = list(enumerate(args_embed))
                known = [('fn',fn_embed)] + labelled_args
                #assert torch.allclose(sing.model.apply_nn(known, target='out'),sing.model.apply_nn(known, target='out')) # TODO remove
                return Examplewise(abstract=sing.model.apply_nn(known, target='out'))

            else:
                raise TypeError
        
        res = self.pnode_cache.beval(self,no_cache,ctx)
        return printed(res)


    def embed_from_above(self):
        """
        Imagine youre a hole. What would multidirectional propagation say the representation
        of you is?
        Note that this does NOT include your type or your context (beyond the fact that
        context is used other places)
        """
        sing.scratch.beval_print(f'embed_from_above {self.marked_str()} ')

        root = self.root()
        zipper = self.get_zipper()
        res = root.inverse_beval(ctx=None, output_ew=None, zipper=zipper)
        return res

    def __eq__(self,other):
        return self is other

    def inverse_beval(self, ctx, output_ew, zipper):
            """
            follow `zipper` downward starting at `self` 
            """
            sing.scratch.beval_print(f'inverse_beval {self} with zipper={zipper} and ctx={ctx}', indent=True)
        
            if hasattr(self,'needs_ctx'):
                assert self.ntype.hole, "temp"
                sing.scratch.beval_print('[hit needs_ctx]')
                assert self.needs_ctx is None, "someone forgot to garbage collect"
                self.needs_ctx = ctx

            def printed(res):
                assert res is not None
                sing.scratch.beval_print(f'{mlb.mk_green("->")} {short_repr(res)}', dedent=True)
                return res

            if len(zipper) == 0:
                """
                reached end of zipper so we can just return the embedding
                without even looking at what node we're pointing to
                """
                sing.scratch.beval_print(f'[end of zipper]')
                return printed(output_ew)
            
            if self.ntype.output:
                """
                set up the ctx right, set output_ew to task.outputs, and strip off the abstractions
                
                caching: no need. No new Context or EW objects get created here
                """
                assert zipper[0] == 'tree'
                zipper = zipper[1:]
                assert ctx is None
                assert output_ew is None

                body,i = self.tree.unwrap_abstractions(count=True)
                assert len(self.task.ctx) == i
                assert len(zipper) >= i, "zippers must pass all the way thru the toplevel abstraction"
                assert all(x == 'body' for x in zipper[:i])
                zipper = zipper[i:]

                return printed(body.inverse_beval(
                    ctx=self.task.ctx,
                    output_ew=self.task.outputs,
                    zipper=zipper
                    ))

            assert ctx is not None
            assert output_ew is not None

            """
            no children to invert into for these types
            """
            if self.ntype.prim:
                assert False
            elif self.ntype.var:
                assert False
            elif self.ntype.hole:
                assert False

            elif self.ntype.abs:
                """
                We will pass our own ctx into the body since this is the enclosed_ctx anyways since
                    rn we're at definition time not application time.
                We will add placeholders for the lambda args to our context so they can be encoded
                    and referenced by anyone.
                Then we simply do the inverse of the body

                caching: yes, since a new Context is created this will invalidate everyones cache if we dont reuse our existing Context instead
                """
                assert zipper[0] == 'body'

                    # sing.scratch.beval_print(f'[trivial abs cache hit]')

                def no_cache():
                    # no cache hit
                    return EWContext.get_placeholders(self.argc) + ctx
                
                new_ctx = self.pnode_cache.inverse_abs(self, no_cache, ctx, output_ew)
                
                res = self.body.inverse_beval(new_ctx, output_ew, zipper[1:])
                
                return printed(res)

            elif self.ntype.app:
                """
                Applications can be inverted towards and arg (zipper[0] is an int)
                or towards the function (zipper[0] == 'fn')
                """
                if zipper[0] == 'fn':
                    assert False, "not a v1 thing and cant ever show up in v1 anyways"
                
                assert isinstance(zipper[0],int)
                assert 0 <= zipper[0] < len(self.xs)

                """
                Abstract inversion! Bc theres no concrete inversion.
                Beval and abstract the fn and args except the one we're inverting into.
                label the args and run apply_nn with output_ew as the parent vector
                """

                assert self.fn.ntype.prim, "feel free to remove post V1"

                sing.scratch.beval_print(f'[inverting application]')

                sing.scratch.beval_print(f'[beval fn]')
                fn_embed = self.fn.beval(ctx)
                
                sing.scratch.beval_print(f'[beval {len(self.xs)-1} args]')
                labelled_args = [(i,arg.beval(ctx)) for i,arg in enumerate(self.xs) if i!=zipper[0]]

                
                def no_cache():
                    sing.scratch.beval_print(f'[get_abstract calls]') # these dont come until here bc of caching
                    fn_embed_vec = fn_embed.get_abstract()
                    labelled_args_vecs = [(i,arg.get_abstract()) for i,arg in labelled_args]
                    output_ew_vec = output_ew.get_abstract()
                    known = [('fn',fn_embed_vec), ('out',output_ew_vec)] + labelled_args_vecs

                    sing.scratch.beval_print(f'[apply_nn]')
                    return Examplewise(abstract=sing.model.apply_nn(known, target=zipper[0]))
                
                new_output_ew = self.pnode_cache.inverse_app(self,no_cache,output_ew,fn_embed,labelled_args,zipper[0])

                res = self.xs[zipper[0]].inverse_beval(ctx, output_ew=new_output_ew, zipper=zipper[1:])
                return printed(res)

            else:
                raise TypeError


    def size(self):
        """
        gets size of tree below this node
        """
        if self.ntype.output:
            return self.tree.size() # no cost
        elif self.ntype.abs:
            return self.body.size() # no cost
        elif self.ntype.app:
            return self.fn.size() + sum(x.size() for x in self.xs) # sum of fn and arg sizes
        elif self.ntype.var or self.ntype.hole or self.ntype.prim or self.ntype.exwise:
            return 1 # base case
        else:
            raise TypeError
    def depth_of_node(self):
        """
        gets depth of this node below the output node
        """
        if self.ntype.output:
            return 0 # zero cost when it's the output node
        elif self.ntype.abs:
            return self.parent.depth_of_node() # no cost
        elif self.ntype.var or self.ntype.hole or self.ntype.prim or self.ntype.app or self.ntype.exwise:
            return self.parent.depth_of_node() + 1 # parent depth + 1
        else:
            raise TypeError
    @property
    def has_holes(self):
        """
        check if we have any holes
        """
        if self.ntype.output:
            return self.tree.has_holes
        elif self.ntype.abs:
            return self.body.has_holes
        elif self.ntype.var or self.ntype.prim or self.ntype.exwise:
            return False
        elif self.ntype.hole:
            return True
        elif self.ntype.app:
            return self.fn.has_holes or any(x.has_holes for x in self.xs)
        else:
            raise TypeError
    def depth(self):
        """
        gets depth of tree below this node
        """
        if self.ntype.output:
            return self.tree.depth() # no cost
        elif self.ntype.abs:
            return self.body.depth() # no cost
        elif self.ntype.var or self.ntype.hole or self.ntype.prim or self.ntype.exwise:
            return 1 # base case
        elif self.ntype.app:
            return max([x.depth() for x in (*self.xs,self.fn)]) # max among fn and args
        else:
            raise TypeError
    def get_hole(self, ordering, tiebreaking):
        """
        returns a single hole or None if there are no holes in the subtree
        """
        if self.ntype.output:
            return self.tree.get_hole(ordering, tiebreaking)
        elif self.ntype.abs:
            return self.body.get_hole(ordering, tiebreaking)
        elif self.ntype.var or self.ntype.prim or self.ntype.exwise:
            return None
        elif self.ntype.hole:
            return self
        elif self.ntype.app:
            holes = [self.fn.get_hole(ordering, tiebreaking)]+ [x.get_hole(ordering,tiebreaking) for x in self.xs]
            holes = [h for h in holes if h is not None]
            if len(holes) == 0:
                return None

            options = {
                'left': holes[0],
                'right': holes[-1],
                'random': random.choice(holes),
            }

            # common cases
            if ordering in options:
                return options[ordering]


            # check for a depth based tie in which case use tiebreaker
            depths = [h.depth_of_node() for h in holes]
            if all(depth==depths[0] for depth in depths):
                return options[tiebreaking]

            # normal depth based ones
            if ordering == 'deep':
                return max(holes, key=lambda h: h.depth_of_node())
            if ordering == 'shallow':
                return max(holes, key=lambda h: -h.depth_of_node())

            raise ValueError(ordering)
        else:
            raise TypeError
    def children(self,recursive=False):
        """
        returns a list of any nodes immediately below this one in the tree (empty list if leaf)
        note that this doesnt recursively get everyone, just your immediate children.
        """
        if self.ntype.output:
            res = [self.tree]
        elif self.ntype.abs:
            res =  [self.body]
        elif self.ntype.app:
            res = [self.fn,*self.xs]
        elif self.ntype.var or self.ntype.hole or self.ntype.prim or self.ntype.exwise:
            res =  []
        else:
            raise TypeError

        if recursive:
            res += list(itertools.chain.from_iterable(x.children(recursive) for x in res))
        return res
    def get_prod(self):
        self = self.unwrap_abstractions()

        ntype = self.ntype
        if ntype.hole and self.phantom_ntype is not None:
            ntype = self.phantom_ntype

        if ntype.output:
            raise TypeError
        elif ntype.hole:
            raise TypeError
        elif ntype.app:
            return self.fn.get_prod()
        elif ntype.prim:
            return self.prim
        elif ntype.exwise:
            return self.ew
        elif ntype.var:
            return Index(self.i)
        elif ntype.abs:
            assert False, "not possible bc we unwrapped abstractions"
        else:
            raise TypeError
    def unwrap_abstractions(self, count=False):
        """
        traverse down .body attributes until you get to the first non-abstraction FPNode.
        """
        node = self
        i = 0
        while node.ntype.abs:
            node = node.body
            i += 1
        if count:
            return node,i
        return node
    def root(self):
        """
        get the root of the tree (an output node)
        """
        if self.parent is self:
            return self
        return self.parent.root()
    def get_zipper(self):
        """
        Get a zipper to yourself from your self.root()

        Zipper looks like ('tree','body','body',1,2,1) for example where ints are used for which arg of a fn ur talking about
        """
        root = self.root()
        if self is root:
            return ()
        parent = self.parent
        if parent.ntype.output:
            attr = 'tree'
        elif parent.ntype.abs:
            attr = 'body'
        elif parent.ntype.hole or parent.ntype.prim or parent.ntype.var or self.ntype.exwise:
            raise TypeError
        elif parent.ntype.app:
            if parent.fn is self:
                attr = 'fn'
            for i,arg in enumerate(parent.xs):
                if arg is self:
                    attr = i
        elif parent.ntype.abs:
            attr = 'body'
        elif parent.ntype.abs:
            attr = 'body'
        else:
            raise TypeError
        
        return (*parent.get_zipper(),attr)
    def apply_zipper(self, zipper):
        """
        Returns the node retrieved by the zipper
        """
        if len(zipper) == 0:
            return self
        if isinstance(zipper[0],str):
            return getattr(self,zipper[0]).apply_zipper(zipper[1:])
        elif isinstance(zipper[0],int):
            return self.xs[zipper[0]].apply_zipper(zipper[1:])
        else:
            raise TypeError

    def clone(self):
        """
        shallow clones but doesnt dup Exwise objects just shallow copies them.
        """
        zipper = self.get_zipper() # so we can find our way back to this node in the new tree
        root = self.root()

        cloned_root = FPNode.from_ptask(root.task) # share same ptask (includes cache)
        cloned_root.expand_from(root)

        cloned_self = cloned_root.apply_zipper(zipper)
        assert self.marked_str() == cloned_self.marked_str()
        return cloned_self

  