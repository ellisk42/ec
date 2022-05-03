import dreamcoder as dc
from dreamcoder.domains.quantum_algorithms.primitives import *


class QuantumTask(dc.task.Task):
    last_algorithm = ""
    last_algorithm_evaluations = {}

    min_size = 3
    max_size = 8 # exclusive bound

    def __init__(self, name, target_algorithm):
        self.target_algorithm = target_algorithm
        self.target_algorithm_evaluations = {n: full_circuit_to_mat(target_algorithm(n)) for n in range(self.min_size, self.max_size)}

        super(QuantumTask, self).__init__(name=name,
                                          request=dc.type.arrow(tcircuit, tcircuit),
                                          examples=[],
                                          features=[])

    def logLikelihood(self, e, timeout=None):
        if QuantumTask.last_algorithm is not e:
            QuantumTask.last_algorithm = e
            QuantumTask.last_algorithm_evaluations = {}

        for n in range(self.min_size, self.max_size):
            if n not in QuantumTask.last_algorithm_evaluations.keys():
                QuantumTask.last_algorithm_evaluations[n] = execute_quantum_algorithm(e, n, timeout)

            yh = QuantumTask.last_algorithm_evaluations[n]
            yh_true = self.target_algorithm_evaluations[n]

            if yh is None:
                return dc.utilities.NEGATIVEINFINITY
            
            if not np.all(np.abs(yh-yh_true)<= 1e-5):
                return dc.utilities.NEGATIVEINFINITY
                
        return 0.


def makeTasks():

    tasks = [
        QuantumTask("hadamard_0", lambda n_qubit: (n_qubit, (("hadamard", 0),))),
        QuantumTask("cnot_01", lambda n_qubit: (n_qubit, (("cnot", 0, 1),))),
        QuantumTask("cnot_10", lambda n_qubit: (n_qubit, (("cnot", 1,0),))),
        QuantumTask("cnot_02", lambda n_qubit: (n_qubit, (("cnot", 0, 2),))),
        QuantumTask("cnot_20", lambda n_qubit: (n_qubit, (("cnot", 2, 0),))),
        QuantumTask("swap_01", lambda n_qubit: (n_qubit, (("swap", 0, 1),))),
        QuantumTask("swap_02", lambda n_qubit: (n_qubit, (("swap", 0, 2),))),
        QuantumTask("swap_12", lambda n_qubit: (n_qubit, (("swap", 1, 2),))),
        QuantumTask("cz_01", lambda n_qubit: (n_qubit, (("cz", 0, 1),))),
        QuantumTask("cz_12", lambda n_qubit: (n_qubit, (("cz", 1, 2),))),
        QuantumTask("cz_02", lambda n_qubit: (n_qubit, (("cz", 0, 2),))),
        QuantumTask("hadamard_n", lambda n_qubit: (n_qubit, (("hadamard", n_qubit-1),))),
        QuantumTask("hadamard_n_1", lambda n_qubit: (n_qubit, (("hadamard", n_qubit-2),))),
        QuantumTask("cnot_nn_1", lambda n_qubit: (n_qubit, (("cnot", n_qubit-2, n_qubit-1),))),
        QuantumTask("swap_nn_1", lambda n_qubit: (n_qubit, (("swap", n_qubit-2, n_qubit-1),))),
        QuantumTask("cz_nn_1", lambda n_qubit: (n_qubit, (("cz", n_qubit-2, n_qubit-1),))),
        QuantumTask("swap_0n", lambda n_qubit: (n_qubit, (("swap", 0, n_qubit-1),))),
        QuantumTask("swap_0n_1", lambda n_qubit: (n_qubit, (("swap", 0, n_qubit-2),))),
        QuantumTask("cnot_0n", lambda n_qubit: (n_qubit, (("cnot", 0, n_qubit-1),))),
    ]
    return tasks


def get_task_from_name(name, tasks):
    for task in tasks:
        if task.name == name:
            return task
    else:
        raise Exception("Task not found")
