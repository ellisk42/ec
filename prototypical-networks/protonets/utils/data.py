import protonets.data.geometry

def load(opt, splits):
    # if opt['data.dataset'] == 'omniglot':
        # ds = protonets.data.omniglot.load(opt, splits)
    if opt['data.dataset'] == 'geometry':
        ds = protonets.data.geometry.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
