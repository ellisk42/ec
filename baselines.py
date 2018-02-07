"Program learning baselines. All functions take the same arguments as ec."
import ec

def all(*args, **kwargs):
    return {
        "robustfill": robustfill(*args, **kwargs),
        "iterative_pcfg": iterative_pcfg(*args, **kwargs),
        "enumeration": enumeration(*args, **kwargs),
    }


def robustfill(*args, **kwargs):
    kwargs.update({
        "message": "robustfill",
        "onlyBaselines": False,
        "outputPrefix": None,

        "useRecognitionModel": True,
        "iterations": kwargs["iterations"], # XXX: should we change this?
        "helmholtzRatio": 1.0,
        "pseudoCounts": 0,
        "aic": float("inf"),
    })
    return ec.explorationCompression(*args, **kwargs)


def iterative_pcfg(*args, **kwargs):
    kwargs.update({
        "message": "iterative_pcfg",
        "onlyBaselines": False,
        "outputPrefix": None,

        "useRecognitionModel": False,
        "iterations": kwargs["iterations"], # XXX: should we change this?
        "aic": float("inf"),
        "pseudoCounts": 0,
    })
    return ec.explorationCompression(*args, **kwargs)


def enumeration(*args, **kwargs):
    kwargs.update({
        "message": "enumeration",
        "onlyBaselines": False,
        "outputPrefix": None,

        "useRecognitionModel": False,
        "iterations": 1,
        # We will be evaluating the baselines using benchmarking on the testing set
        # So we should just use whatever frontier size will be used for benchmarking
        #"frontierSize": 200000,
    })
    return ec.explorationCompression(*args, **kwargs)
