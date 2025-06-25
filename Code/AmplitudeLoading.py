"""
Class Amplitude Loading inherit from QuantumCircuit and permits to load arbitrary values f(x) in the amplitude (|1> basis) of a target qubit, depending on the basis configurations of the 'state qubits' (|x>).
"""
# =============================================================================
# Module attributes
# =============================================================================
# Documentation format
__docformat__ = 'NumPy'
# License type (e.g. 'GPL', 'MIT', 'Proprietary', ...)
__license__ = 'Proprietary'
# Status ('Prototype', 'Development' or 'Pre-production')
__status__ = 'Development'
# Version (0.0.x = 'Prototype', 0.x.x = 'Development', x.x.x = 'Pre-production)
__version__ = '0.1.0'
# Authors (e.g. code writers)
__author__ = ('xxx',
              'yyy')
# Maintainer
__maintainer__ = ''
# Email of the maintainer
__email__ = ''

# =============================================================================
# Import modules
# =============================================================================
# Import general purpose module(s)

from typing import List
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit.library import UCRYGate, CRYGate, RYGate, CRXGate, RXGate


class AmplitudeLoading(QuantumCircuit):
    """
    xxx
    """
    
    def __init__(self, num_state_qubits: int, scaled_values: List[float], name: str = "f(x)") -> None:
        """
        Args:
            num_state_qubits: the number of state qubits.
            scaled_values: the list of the values in the range [-1, 1]. Its size must be 2^num_state_qubits.
            name: the name of the circuit
        """
        self.num_state_qubits = num_state_qubits
        self.scaled_values = scaled_values
        self.name = name
        
        if (min(scaled_values) < -1) or (max(scaled_values) > 1):
            raise ValueError("The range must be contained in [-1, 1].")

        if num_state_qubits != np.log2(len(scaled_values)):
            raise ValueError("The number of values in input does not match the power of 2 of the number of qubits that encode the state.")

        # The elements of scaled_values are converted to alpha rotations multiplying them to 2 * arcsin
        alpha_values = 2 * np.arcsin(scaled_values)
        
        ctrl_q = QuantumRegister(num_state_qubits, name="State")
        target_q = QuantumRegister(1, name="Target")
        
        super().__init__(ctrl_q, target_q, name=self.name)
        
        self.append(UCRYGate(list(alpha_values)), target_q[:] + ctrl_q[:])
        

class AmplitudeLoadingV2(QuantumCircuit):
    """
    xxx
    """
    
    def __init__(self, num_state_qubits: int, scaled_values: List[float], c: float, name: str = "f(x)") -> None:
        """
        Args:
            num_state_qubits: the number of state qubits.
            scaled_values: the list of the values in the range [-1, 1]. Its size must be 2^num_state_qubits.
            name: the name of the circuit
        """
        self.num_state_qubits = num_state_qubits
        self.scaled_values = scaled_values
        self.name = name
        
        if (min(scaled_values) < -1) or (max(scaled_values) > 1):
            raise ValueError("The range must be contained in [-1, 1].")

        if num_state_qubits != len(scaled_values):
            raise ValueError("The number of values in input does not match the power of 2 of the number of qubits that encode the state.")

        # The elements of scaled_values are converted to alpha rotations multiplying them to 2 * arcsin
        alpha_values = [c*s*2 for s in scaled_values]
        print(f'alphas: {alpha_values}')
        
        ctrl_q = QuantumRegister(num_state_qubits, name="State")
        target_q = QuantumRegister(1, name="Target")
        
        super().__init__(ctrl_q, target_q, name=self.name)
        
        self.append(MultiCRYGateV2(list(alpha_values), ctrl_q[:] + target_q[:], np.pi/2 - c), ctrl_q[:] + target_q[:])
        
    
    
def MultiCRYGateV2(alpha_values: list, q_reg : QuantumRegister, starting_offset:float = 0) :
    circuit = QuantumCircuit(q_reg)
    q_target=q_reg[-1]
    if starting_offset != 0:
        circuit.append(RYGate(starting_offset, label="start_offset"), [q_target])
    # TODO: controllare che il numero degli angoli e dei qubit siano coerenti
    for i in range(0,len(q_reg)-1) :
        cy_gate = CRYGate(alpha_values[i], label="rot_"+str(i))
        circuit.append(cy_gate, [q_reg[i],q_target])
    return circuit    
    
    
def MultiCRGateVar(alpha_values: list, q_reg : QuantumRegister, axis='Y', starting_offset:float = 0) :
    circuit = QuantumCircuit(q_reg)
    q_target=q_reg[-1]
    if axis=='Y':
        rgate = RYGate
        crgate = CRYGate
    else:
        rgate = RXGate
        crgate = CRXGate
        
    if starting_offset != 0:
        circuit.append(rgate(starting_offset, label="start_offset_"+axis), [q_target])
    # TODO: controllare che il numero degli angoli e dei qubit siano coerenti
    for i in range(0,len(q_reg)-1) :
        c_gate = crgate(alpha_values[i], label="rot_"+axis+"_"+str(i))
        circuit.append(c_gate, [q_reg[i],q_target])
    return circuit.to_gate()
        

class AmplitudeLoadingVar(QuantumCircuit):
    """
    Create the circuit that given the state |j>|0> returns 
        |j>(cos(alpha_j)|0> + sin(alpha_j)|1>)
    where alpha_j is an item of a given list (scaled_values).
    
    Args:
        num_state_qubits: number of qubits corresponding to the encoding of states j's
        scaled_values: trigonometric functions argument in target states (scaled_values=[alpha_0, alpha_1, ... ,alpha_{2^num_state_qubits -1}])
        name: given name of the circuit
    """
    
    def __init__(self, num_state_qubits: int, scaled_values: List[float], starting_offset:float = 0, name: str = "f(x)") -> None:
        """
        Args:
            num_state_qubits: the number of state qubits.
            scaled_values: the list of the values in the range [-pi/2, pi/2]. Its size must be 2^num_state_qubits.
            name: the name of the circuit
        """
        self.num_state_qubits = num_state_qubits
        self.scaled_values = scaled_values
        self.name = name
        
        if (min(scaled_values) < -np.pi) or (max(scaled_values) > np.pi):
            raise ValueError("The range must be contained in [-pi, pi].")

        if num_state_qubits != len(scaled_values):
            raise ValueError("The number of values in input does not match the power of 2 of the number of qubits that encode the state.")

        alpha_values = [s*2 for s in scaled_values]
        # print(f'alphas: {alpha_values}')
        
        ctrl_q = QuantumRegister(num_state_qubits, name="State")
        target_q = QuantumRegister(1, name="Target")
        
        super().__init__(ctrl_q, target_q, name=self.name)
        
        self.append(MultiCRGateVar(list(alpha_values), ctrl_q[:] + target_q[:], 'Y', 2*starting_offset), ctrl_q[:] + target_q[:])
        self.x(target_q) # scambio 1 e 0 per avere la codifica (con il seno) sullo 0
