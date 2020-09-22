

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