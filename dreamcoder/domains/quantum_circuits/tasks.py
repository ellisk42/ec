import dreamcoder as dc
from dreamcoder.domains.quantum_circuits.primitives import *
from dreamcoder.domains.quantum_circuits.primitives import circuit_to_mat


class QuantumTask(dc.task.Task):
    last_circuit = ""
    last_circuit_evaluation = {}

    def __init__(self, name, target_circuit):
        self.n_qubits, self.target_circuit = target_circuit
        self.target_circuit_evaluation = circuit_to_mat(target_circuit)
        
        super(QuantumTask, self).__init__(name=name,
                                          request=dc.type.arrow(*([dc.type.tint]*self.n_qubits), tcircuit, tcircuit),
                                          examples=[((*range(self.n_qubits),no_op(self.n_qubits),),(self.target_circuit_evaluation,),)],
                                          features=[])

    def logLikelihood(self, e, timeout=None):
        if QuantumTask.last_circuit is not e:
            QuantumTask.last_circuit = e
            QuantumTask.last_circuit_evaluation = None

        if  QuantumTask.last_circuit_evaluation is None:
            QuantumTask.last_circuit_evaluation = execute_quantum_algorithm(e, self.n_qubits, timeout)

        yh_true = self.target_circuit_evaluation
        
        yh = QuantumTask.last_circuit_evaluation
        # TO DISABLE CACHING (for testing)
        # yh =  execute_quantum_algorithm(e, self.n_qubits, timeout)
        
        if yh is None:
            return dc.utilities.NEGATIVEINFINITY
        
        try:
            # # estimate phase with sum of elements
            # s1 = np.sum(yh)
            # s_true = np.sum(yh_true)
            
            # # if sum of all elements is not a good estimate of phase, take first non zero element
            # if s1 ==0 or s_true==0:
            #     idx = np.where((yh*yh_true).round(3)!=0)
            #     if len(idx[0]) ==0:
            #         if not np.all(np.abs(yh-yh_true)<= 1e-3):
            #             return dc.utilities.NEGATIVEINFINITY
            #         else: return 0
            #     else:
            #         s1 = yh[idx[0][0], idx[1][0]]
            #         s_true = yh_true[idx[0][0], idx[1][0]]

            # # Test unitary equivalence (identity up to a phase)
            # if not np.all(np.abs(yh/s1-yh_true/s_true)<= 1e-3):
            #     return dc.utilities.NEGATIVEINFINITY
            
            # # Without phase equivalence: already implemented when calculating unitary
            
            if np.any(np.abs(yh-yh_true)>= 1e-3):
                return dc.utilities.NEGATIVEINFINITY
        except ValueError:
            return dc.utilities.NEGATIVEINFINITY 
        return 0.

def makeTasks(task_enumeration_timeout=6):
    pcfg_full = dc.grammar.PCFG.from_grammar(full_grammar, request=dc.type.arrow(
                                                                                *[dc.type.tint]*dc.domains.quantum_circuits.primitives.GLOBAL_NQUBIT_TASK,
                                                                                tcircuit, tcircuit))
    tasks = dc.enumeration.enumerate_pcfg(pcfg_full,
                                timeout=task_enumeration_timeout, 
                                observational_equivalence=True,
                                sound=True)
    
    quantumTasks = []
    for idx, task in enumerate(tasks.values()):
        quantumTasks.append(QuantumTask(f"t_{idx:03d}_{task['code']}", task["circuit"]))
    return quantumTasks


def get_task_from_name(name, tasks):
    for task in tasks:
        if task.name == name:
            return task
    else:
        raise Exception("Task not found")
