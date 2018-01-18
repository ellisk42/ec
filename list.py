from ec import *
from utilities import eprint
from json_tasks import load_json_tasks_from_file
from listPrimitives import primitives


if __name__ == "__main__":
    tasks = load_json_tasks_from_file("data/list_tasks.json", use_test = True)
    eprint("Got {0} list tasks".format(len(tasks)))

    explorationCompression(primitives, tasks,
                           outputPrefix = "experimentOutputs/list",
                           **commandlineArguments(frontierSize = 10**4,
                                                  a = 1,
                                                  iterations = 10,
                                                  pseudoCounts = 10.0))

