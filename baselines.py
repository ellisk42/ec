"Program learning baselines. All functions take the same arguments as ec."
import ec

def all(*args, **kwargs):
    return {
        "robustfill": robustfill(*args, **kwargs),
        "iterative_pcfg": iterative_pcfg(*args, **kwargs),
        "ec_v1": ec_v1(*args, **kwargs),
        # Enumeration is equivalent to the first wake cycle
        # We already have this data in results.testingTimes[0]
#        "enumeration": enumeration(*args, **kwargs),
    }


def robustfill(*args, **kwargs):
    kwargs.update({
        "message": "robustfill",
        "onlyBaselines": False,
        "outputPrefix": None,

        # Training robust fill doesn't use any enumeration
        # So we just put this to a small value
        "enumerationTimeout": 1,

        "useRecognitionModel": True,
        # Trained a recognition model on a ton of iterations on only samples from an unlearned generative model
        "iterations": 1,
        "steps": 50000,
        "helmholtzRatio": 1.0,
        "pseudoCounts": 1.,
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

def ec_v1(*args, **kwargs):
    kwargs.update({
        "message": "ECv1",
        "onlyBaselines": False,
        "outputPrefix": None,

        "useRecognitionModel": False,
        "iterations": kwargs["iterations"], # XXX: should we change this?
        "pseudoCounts": 0.1,
        "arity": 0,
        "maximumFrontier": 2,
        "topK": 2
    })
    return ec.explorationCompression(*args, **kwargs)


def enumeration(*args, **kwargs):
    kwargs.update({
        "message": "enumeration",
        "onlyBaselines": False,
        "outputPrefix": None,

        "useRecognitionModel": False,
        "iterations": 1,
        "aic": float("inf"),
        # We will be evaluating the baselines using benchmarking on the testing set
        # So we should just use whatever frontier size will be used for benchmarking
        #"frontierSize": 200000,
    })
    return ec.explorationCompression(*args, **kwargs)
