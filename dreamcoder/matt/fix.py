
from omegaconf import DictConfig,OmegaConf,open_dict

def fix_cfg(cfg):
    if 'is_dirty' not in cfg:
        with open_dict(cfg):
            cfg.is_dirty = True




def fix_state_testmode(state):
    if not hasattr(state.phead,'featureExtractor'):
        return state
    ext = state.phead.featureExtractor
    if not hasattr(ext,'lexicon_embedder'):
        ext.lexicon_embedder = ext.digit_embedder
        lex = ext.lexicon_embedder
        lex.ctx_start = lex.idx_of_tok['CTX_START']
        lex.ctx_end = lex.idx_of_tok['CTX_END']
        lex.int_start = lex.idx_of_tok['INT_START']
        lex.int_end = lex.idx_of_tok['INT_END']
        lex.list_start = lex.idx_of_tok['LIST_START']
        lex.list_end = lex.idx_of_tok['LIST_END']
    if not hasattr(state.cfg.model,'ctxful_lambdas'):
        with  open_dict(state.cfg.model):
            state.cfg.model.ctxful_lambdas = False
    return state



def fix_model_result(m):
    from dreamcoder.matt.plot import ModelResult # needs to go in here to avoid cyclic import error
    new = ModelResult(
        prefix='prefix',
        name=m.name,
        cfg=None,
        search_results=m.search_results,
        search_failures=[],
        timeout=m.max_time
    )
    new.num_tests = m.num_tests
    return new