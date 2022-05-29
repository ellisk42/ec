
import numpy as np
import dreamcoder as dc
from dreamcoder.utilities import eprint

try:
    import matplotlib.pyplot as plt
    import qiskit as qk
    from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit,Aer
    backend = Aer.get_backend('unitary_simulator')
    from qiskit.transpiler.synthesis import solovay_kitaev
    skd = solovay_kitaev.SolovayKitaevDecomposition()
    
    # ---------------------------------------------------------------------------------
    # Transpiler configuration
    basis_gates=['h',"cx",'t',"tdg"] 

    ## Qiskit implementation, which natively also includes plotting 
    class QiskitTester():
        def __init__(self,n_qubits=None,circuit=None):
            self.n_qubits = n_qubits
            self.qreg_q = QuantumRegister(self.n_qubits, 'q')
            self.circuit = QuantumCircuit(self.qreg_q)
            if circuit!=None:
                self.unitary_matrix = circuit_to_mat(circuit) # full_circuit
                self.unitary_tensor = mat_to_tensor(self.unitary_matrix)
                self.n_qubits = get_qubit_number(self.unitary_tensor)
                if self.n_qubits!=n_qubits:
                    raise Exception("Duplicate inconsistent arguments!")
                
        def q(self,q_num):
            return self.n_qubits -1- q_num
        
        def __enter__(self):
            return self
        
        def get_result(self, circuit):
            return np.array(qk.execute(circuit, backend).result().get_unitary()).T
        
        def __exit__(self,*args, **kwargs):
            self.result = self.get_result(self.circuit)
            
        def __str__(self) -> str:
            return self.circuit.__str__()
            
        def check(self):
            # Checks that unitary code is consistent with what Qiskit would generate
            try:
                np.testing.assert_almost_equal(self.unitary_matrix,self.result, decimal=3)
                eprint("Code consistent with Qiskit")
            except AssertionError as e:
                eprint("-----------------------------------")
                eprint("ERROR: ")
                eprint(self.unitary_matrix)
                eprint(self.result)
                eprint(e)
                
        def get_transpiled(self, circuit):
            transpiled=qk.transpile(circuit, backend)
            circuit2 = pm.run(transpiled)
            discretized = skd(circuit2)
            return qk.transpile(discretized,backend, basis_gates)
        
        def transpile(self):
            return self.get_transpiled(self.circuit)
        

    def print_circuit(full_circuit, filename=None):
        with QiskitTester(full_circuit) as QT:
            n_qubit, op_list = full_circuit
            op_names = qiskit_full_op_names(QT)
            for op in op_list:
                op_names[op[0]](*op[1:])
                
            # pip install pylatexenc for mpl draw
            QT.circuit.draw(output="mpl", filename=filename) if filename is not None else eprint(QT) 
            plt.show()
            

    with QiskitTester(1) as QT:
        QT.circuit.t(0)
        QT.circuit.t(0)
    qk.circuit.equivalence_library.StandardEquivalenceLibrary.add_equivalence(qk.circuit.library.SGate(),QT.circuit)

    with QiskitTester(1) as QT:
        QT.circuit.tdg(0)
        QT.circuit.tdg(0)
    qk.circuit.equivalence_library.StandardEquivalenceLibrary.add_equivalence(qk.circuit.library.SdgGate(),QT.circuit)

    
    class ParametricSubstitution(qk.transpiler.TransformationPass):
        def run(self, dag):
            # iterate over all operations
            for node in dag.op_nodes():
                print(node.op.name, node.op.params)
                # if we hit a RYY or RZZ gate replace it
                
                if node.op.name in ["cp"]:
                    replacement = QuantumCircuit(2)
                    replacement.p(node.op.params[0]/2,0)
                    replacement.cx(0,1)
                    replacement.p(-node.op.params[0]/2,1)
                    replacement.cx(0,1)
                    replacement.p(node.op.params[0]/2,1)

                    # replace the node with our new decomposition
                    dag.substitute_node_with_dag(node, qk.converters.circuit_to_dag(replacement))
                                    
                
                if node.op.name in ["p"] and node.op.params[0]==np.pi/2:

                    # calculate the replacement
                    replacement = QuantumCircuit(1)
                    replacement.s([0])

                    # replace the node with our new decomposition
                    dag.substitute_node_with_dag(node, qk.converters.circuit_to_dag(replacement))
                    
                elif node.op.name in ["p"] and node.op.params[0]==3*np.pi/2:
        
                    # calculate the replacement
                    replacement = QuantumCircuit(1)
                    replacement.tdg([0])
                    replacement.tdg([0])

                    # replace the node with our new decomposition
                    dag.substitute_node_with_dag(node, qk.converters.circuit_to_dag(replacement))
                                
                elif node.op.name in ["p"] and node.op.params[0]==5*np.pi/2:
            
                    # calculate the replacement
                    replacement = QuantumCircuit(1)
                    replacement.t([0])
                    replacement.t([0])

                    # replace the node with our new decomposition
                    dag.substitute_node_with_dag(node, qk.converters.circuit_to_dag(replacement))
                                
                                
            return dag
    pm = qk.transpiler.PassManager()
    pm.append(ParametricSubstitution())

    # ------------------------------------------ End of Qiskit code
except:
    eprint("Qiskit not found. Necessary for quantum circuit plots.")
    
class QuantumCircuitException(Exception):
    ...
    
# ------------------------------------------
# Define functions that operate on unitaries
# (written as tensors)


def mat_to_tensor(mat):
    n_qubits = int(np.math.log2(mat.shape[0]))
    return mat.reshape([2]*n_qubits*2)
    # first qubits are output, last qubits are input


def get_qubit_number(tensor):
    return int(len(tensor.shape)/2)


def tensor_to_mat(tensor):
    dim_space = 2**get_qubit_number(tensor)
    return tensor.reshape([dim_space, dim_space])


def tensor_product(A, B):
    shape = A.shape[-1]*B.shape[-1]
    return np.reshape(A[..., :, None, :, None]*B[..., None, :, None, :], (-1, shape, shape))


def tensor_contraction(A, B, indices):
    n_qubits = get_qubit_number(A)
    idx = [i + n_qubits for i in indices]
    out = np.tensordot(A, B, (idx, np.arange(len(indices))))
    return np.moveaxis(out, np.arange(-len(indices), 0, 1), idx)

# not really needed, just for completeness


def mat_contraction(A, B, indices):
    return tensor_to_mat(tensor_contraction(
        mat_to_tensor(A),
        mat_to_tensor(B),
        indices)
    )

# ------------------------------------------
# Some simple gates
#

mat_cnot = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=np.float16)
tensor_cnot = mat_to_tensor(mat_cnot)

mat_swap = np.array([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]], dtype=np.float16)
tensor_swap = mat_to_tensor(mat_swap)


mat_hadamard = np.array([[1,1],
                         [1,-1]], dtype=np.float16)/np.sqrt(2)
tensor_hadamard = mat_to_tensor(mat_hadamard)

mat_t = np.array([[1,0],
                  [0,(1+1j)/np.sqrt(2)]])
tensor_t = mat_to_tensor(mat_t)

mat_tdg =np.array([[1,0],
                  [0,(1-1j)/np.sqrt(2)]])
tensor_tdg = mat_to_tensor(mat_tdg)

mat_x = np.array([[0,1],
                  [1,-0]], dtype=np.float16)
tensor_x = mat_to_tensor(mat_x)

mat_y = np.array([[0,1j],
                  [1j,0]])
tensor_y = mat_to_tensor(mat_y)

mat_z = np.array([[1,0],
                  [0,-1]], dtype=np.float16)
tensor_z = mat_to_tensor(mat_z)

mat_s = np.array([[1,0],
                  [0,1j]])
tensor_s = mat_to_tensor(mat_s)

mat_sx = np.array([[1+1j,1-1j],
                  [1-1j,1+1j]])/2
tensor_sx = mat_to_tensor(mat_sx)

mat_sxdg = np.array([[1-1j,1+1j],
                  [1+1j,1-1j]])/2
tensor_sxdg = mat_to_tensor(mat_sxdg)




mat_cy = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1j],
                     [0,0,-1j,0]])
tensor_cy = mat_to_tensor(mat_cy)


mat_cz = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,-1]], dtype=np.float16)
tensor_cz = mat_to_tensor(mat_cz)

mat_cs = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1j]])
tensor_cs = mat_to_tensor(mat_cs)


mat_iswap = np.array([[1,0,0,0],
                     [0,0,1j,0],
                     [0,1j,0,0],
                     [0,0,0,1]])
tensor_iswap = mat_to_tensor(mat_iswap)

# ------------------------------------------
# Apply a gate on all given qubits
#
def eye(n):
    return mat_to_tensor(np.eye(2**n, dtype=np.float16))


def hadamard(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_hadamard, [qubit_1])

def t(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_t, [qubit_1])

def tdg(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_tdg, [qubit_1])

def s(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_s, [qubit_1])

def x(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_x, [qubit_1])

def y(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_y, [qubit_1])

def z(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_z, [qubit_1])

def sx(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_sx, [qubit_1])

def sxdg(circuit, qubit_1):
    return tensor_contraction(circuit, tensor_sxdg, [qubit_1])


def cnot(circuit, qubit_1, qubit_2):
    return tensor_contraction(circuit, tensor_cnot, [qubit_1, qubit_2])

def cy(circuit, qubit_1, qubit_2):
    return tensor_contraction(circuit, tensor_cy, [qubit_1, qubit_2])

def cz(circuit, qubit_1, qubit_2):
    return tensor_contraction(circuit, tensor_cz, [qubit_1, qubit_2])

def cs(circuit, qubit_1, qubit_2):
    return tensor_contraction(circuit, tensor_cs, [qubit_1, qubit_2])


def swap(circuit, qubit_1, qubit_2):
    return tensor_contraction(circuit, tensor_swap, [qubit_1, qubit_2])

def iswap(circuit, qubit_1, qubit_2):
    return tensor_contraction(circuit, tensor_iswap, [qubit_1, qubit_2])


# ------------------------------------------
# Transform a list of qubit operations into a unitary 
#
full_op_names = {
    # "eye": eye,
    "hadamard": hadamard,
    "t": t,
    "tdg": tdg,
    "s": s,
    "x": x,
    "y": y,
    "z": z,
    "sx": sx,
    "sxdg": sxdg,
    "cnot": cnot,
    "cy": cy,
    "cz": cz,
    "cs": cs,
    "swap": swap,
    "iswap":iswap
}

qiskit_full_op_names = lambda QT: {
    "eye": lambda q1: QT.circuit.id(QT.q(q1)),
    "hadamard": lambda q1: QT.circuit.h(QT.q(q1)),
    "cnot":  lambda q1,q2: QT.circuit.cnot(QT.q(q1),QT.q(q2)),
    "swap": lambda q1,q2: QT.circuit.swap(QT.q(q1),QT.q(q2)),
    "cz": lambda q1,q2: QT.circuit.cz(QT.q(q1),QT.q(q2)),
}

eyes = {} #caching initial identity matrices
full_circuit_cache = {}
def circuit_to_mat(full_circuit):
    eprint(full_circuit)
    t_full_circuit = tuple(full_circuit)
    try:
        if t_full_circuit not in full_circuit_cache:
            n_qubit, op_list = full_circuit
            
            if n_qubit not in eyes.keys():
                eyes[n_qubit]=eye(n_qubit)
            tensor = eyes[n_qubit]
            
            for op in op_list:
                tensor = full_op_names[op[0]](tensor, *op[1:])
                
            full_circuit_cache[t_full_circuit] = tensor_to_mat(tensor)
    except TypeError as e:
        print(e)
        ...
    return full_circuit_cache[t_full_circuit]



# only for testing
def qiskit_circuit_to_mat(full_circuit):
    t_full_circuit = tuple(full_circuit)
    try:
        if t_full_circuit not in full_circuit_cache:
            n_qubit, op_list = full_circuit
            
            with QiskitTester(n_qubit) as QT:
                op_names = qiskit_full_op_names(QT)
                for op in op_list:
                    op_names[op[0]](*op[1:])
                
            full_circuit_cache[t_full_circuit] = QT.result
    except TypeError as e:
        ...
    return full_circuit_cache[t_full_circuit]




# ------------------------------------------
# Define functions for primitives (to act on circuits)
#
# tsize = dc.type.baseType("tsize")
tcircuit = dc.type.baseType("tcircuit")


# Control
def _repeat(n_times,body):
    if n_times <= 0:
        raise QuantumCircuitException("Invalid repetition number.")
    return   _repeat_help(n_times, body, body)

def _repeat_help(n_times, body, new_body):
    if n_times==1:
        return new_body
    return _repeat_help(n_times-1, body, lambda b: new_body(body(b)))




# ------------------------------------------
# Define FULL primitives
# 

# Arithmetics
p_0 = dc.program.Primitive("0",dc.type.tint,0)
p_inc = dc.program.Primitive("inc", dc.type.arrow(dc.type.tint, dc.type.tint), lambda x:x+1)
p_dec = dc.program.Primitive("dec", dc.type.arrow(dc.type.tint, dc.type.tint), lambda x:x-1)

## Full circuit [n_qubits, [ops]]
def no_op(n):
    return (n, ())

def get_n_qubits(old_circuit):
    return old_circuit[0]

def one_qubit_gate(old_circuit, qubit_1,operation_name):
    n_qubit, circuit = old_circuit
        
    if qubit_1<0 or qubit_1 >= n_qubit:
        raise QuantumCircuitException("Invalid selected qubit")
    
    circuit = circuit + ((operation_name, qubit_1),)
    return (n_qubit, circuit)

def two_qubit_gate(old_circuit, qubit_1, qubit_2, operation_name):
    # operation_name = "cnot" or some other gate name
    n_qubit, circuit = old_circuit
    
    if qubit_1<0 or qubit_1 >= n_qubit:
        raise QuantumCircuitException("Invalid selected qubit")
    
    if qubit_2<0 or qubit_2 >= n_qubit:
        raise QuantumCircuitException("Invalid selected qubit")
   
    if qubit_1 == qubit_2:
        raise QuantumCircuitException("Invalid selected qubit")
    
    circuit = circuit + ((operation_name, qubit_1, qubit_2),)
    return (n_qubit, circuit)

# Circuit primitives

p_size = dc.program.Primitive(name="size", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint),
                     value=get_n_qubits)

p_hadamard = dc.program.Primitive(name="h", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "hadamard")))

p_t = dc.program.Primitive(name="t", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "t")))

p_tdg = dc.program.Primitive(name="tdg", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "tdg")))

p_s = dc.program.Primitive(name="s", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "s")))

p_sx = dc.program.Primitive(name="sx", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "sx")))

p_sxdg = dc.program.Primitive(name="sxdg", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "sxdg")))



p_x = dc.program.Primitive(name="x", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "x")))

p_y = dc.program.Primitive(name="y", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "y")))

p_z = dc.program.Primitive(name="z", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1: one_qubit_gate(old_circuit, qubit_1, "z")))


p_cnot = dc.program.Primitive(name="cnot", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, dc.type.tint,tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1, qubit_2: two_qubit_gate(old_circuit, qubit_1, qubit_2, "cnot")))

p_cy = dc.program.Primitive(name="cy", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1, qubit_2: two_qubit_gate(old_circuit, qubit_1, qubit_2, "cy")))

p_cz = dc.program.Primitive(name="cz", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1, qubit_2: two_qubit_gate(old_circuit, qubit_1, qubit_2, "cz")))


p_cs = dc.program.Primitive(name="cs", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1, qubit_2: two_qubit_gate(old_circuit, qubit_1, qubit_2, "cs")))

p_swap = dc.program.Primitive(name="swap", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1, qubit_2: two_qubit_gate(old_circuit, qubit_1, qubit_2, "swap")))

p_iswap = dc.program.Primitive(name="iswap", 
                     ty=dc.type.arrow(tcircuit, dc.type.tint, dc.type.tint, tcircuit),
                     value=dc.utilities.Curried(lambda old_circuit, qubit_1, qubit_2: two_qubit_gate(old_circuit, qubit_1, qubit_2, "iswap")))

# Control
p_iteration = dc.program.Primitive(name="rep", 
                     ty=dc.type.arrow(dc.type.tint, dc.type.arrow(tcircuit,tcircuit),  dc.type.arrow(tcircuit,tcircuit)),
                     value=dc.utilities.Curried(_repeat))


full_primitives = [
    #circuits
    p_hadamard,
    p_t,
    p_tdg,
    p_s,
    p_x,
    p_y,
    p_z,
    p_sx,
    p_sxdg,
    p_cnot,
    p_cy,
    p_cz,
    p_cs,
    p_swap,
    p_iswap,
    #arithmetics
    # p_0,
    # p_inc,
    # p_dec,
    # p_size,
    # #control
    # fp_iteration
]

primitives = [
    #circuits
    p_hadamard,
    p_t,
    p_tdg,
    p_cnot,
    #arithmetics
    # p_0,
    # p_inc,
    # p_dec,
    # p_size,
    # #control
    # fp_iteration
]

# ------------------------------------------
# Define GRAMMAR
# 
full_grammar = dc.grammar.Grammar.uniform(full_primitives, continuationType=tcircuit)
grammar = dc.grammar.Grammar.uniform(primitives, continuationType=tcircuit)
            

# ------------------------------------------
# Function to execute algorithms (which are functions)
# Maybe it should return a function?
# 
def execute_quantum_algorithm(p, n_qubits, timeout=None):
    try:
        eprint(p,n_qubits) #it may not have as many inputs?? WHY?
        circuit = p.evaluate([])
        for arg in (*np.arange(n_qubits),no_op(n_qubits)):
            circuit = circuit(arg)
        # circuit =  dc.utilities.runWithTimeout(
        #     lambda: p.evaluate([])(no_op(n_qubits)),
        #     timeout=timeout
        # )  TIMEOUT DISABLED
        return circuit_to_mat(circuit)
    except dc.utilities.RunWithTimeout: return None
    except Exception as e: 
        eprint(e)
        return None
    
    
