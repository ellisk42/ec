try:
    import binutil  # required to import from lib modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from lib.domains.tower.main import main


if __name__ == '__main__':
    main()
