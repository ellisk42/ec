try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.logo.makeLogoTasks import demoLogoTasks

if __name__ == "__main__":
    demoLogoTasks()
