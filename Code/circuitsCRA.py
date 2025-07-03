from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZGate, UnitaryGate, MCMT

import numpy as np

# Import custom module(s)
from Code.utils import mapping
from Code.AmplitudeLoading import AmplitudeLoading, AmplitudeLoadingV2, AmplitudeLoadingVar
from Code.QSVT import QSVT

# =============================================================================
# Create the Expected Loss circuit
# =============================================================================
def get_expected_loss_circuit(K, uncertainity_model, lgd):
    """
    
    """
    # define linear objective function for expected loss
    breakpoints = list(range(0,2**K))
    # per ogni combinazione di assets dà la somma delle perdite
    offsets = [mapping(el, lgd, K) for el in breakpoints]

    # scaling of original values to range [0,1]
    range_min = 0
    range_max = 1
    scaled_offsets = [np.sqrt((off - min(offsets)) / (max(offsets) - min(offsets)) *
                          (range_max - range_min) + range_min) for off in offsets]
    
    objective_e_loss = AmplitudeLoading(K, scaled_offsets)
    
    # define the registers for convenience and readability
    qr_state = QuantumRegister(uncertainity_model.num_qubits, 'state')
    qr_obj = QuantumRegister(1, 'objective')
    ar = QuantumRegister(objective_e_loss.num_ancillas, "work")  # additional qubits

    # define the circuit
    state_preparation = QuantumCircuit(qr_state, qr_obj, ar, name='A')

    # load the random variable
    state_preparation.append(uncertainity_model.to_gate(), qr_state)

    # linear objective function (does aggregation and comparison)
        
    state_preparation.append(objective_e_loss.to_gate(), qr_state[-K:] + qr_obj[:] + ar[:])

    return state_preparation, objective_e_loss

# =============================================================================
# Create the Cumulative distribution function circuit
# =============================================================================
def get_cdf_circuit(K, u, x_eval, lgd):
    
    breakpoints = list(range(0,2**K))
    
    offsets = [1 if mapping(el, lgd, K)<=x_eval else 0 for el in breakpoints]
    
    objective = AmplitudeLoading(K, offsets)

    # define the registers for convenience and readability
    qr_state = QuantumRegister(u.num_qubits, 'state')
    qr_obj = QuantumRegister(1, 'objective')
    ar = QuantumRegister(objective.num_ancillas, "work")  # additional qubits2

    # define the circuit
    state_preparation = QuantumCircuit(qr_state, qr_obj, ar, name='A')

    # load the random variable
    state_preparation.append(u.to_gate(), qr_state)

    state_preparation.append(objective.to_gate(), qr_state[-K:] + qr_obj[:] + ar[:])
    
    return state_preparation, objective


# =============================================================================
# Create the Expected Loss circuit
# =============================================================================

def post_processing(x, lgd) :
    return ( x * sum(lgd) )


def get_expected_loss_circuitV2(K, uncertainity_model, lgd, c):
    """
    
    """
    if c == 0 :
        print('get_expected_loss_circuit standard')
        return get_expected_loss_circuit(K, uncertainity_model, lgd)
    
    print('get_expected_loss_circuitV custom')
    # define linear objective function for expected loss
    breakpoints = list(range(0,2**K))
    offsets = lgd

    # scaling of original values to range [-1,1]
    range_min = -1
    range_max = 1
    minimum=-sum(offsets)
    maximum=sum(offsets)
    scaled_offsets = [off/maximum for off in offsets]
    
    print('scaled offsets: '+str(scaled_offsets))
    
    objective_e_loss = AmplitudeLoadingV2(K, scaled_offsets, c)
    
    return build_state_preparation_circuit(uncertainity_model, objective_e_loss, K), objective_e_loss


def post_processingV2(x, lgd, c=0) :
    if c == 0 :
        return post_processing(x, lgd)
    maximum=sum(lgd)
    return maximum/c*(x-1/2 + c/2)



def get_expected_probability_circuit(
    K: int,
    uncertainity_model: QuantumCircuit,
    lgd: list,
    target_loss: float,
    phases=None,
    poly:list=None,
    threshold=0.5,
    enable_switch=True, epsilon=0.6,
    verbose=False
) -> (QuantumCircuit, QuantumCircuit):
    """
    This function prepares a circuit such that the probability of obtaining 1
        measuring an additional qubit (the last one) equals the probability for
        of having a total loss that is at most target_loss.
    To do that, it applies QSVT with a polynomial that approximates the even
        threshold function (or rectangular function) to a circuit that
        cleaverly encodes all possible scenarios.

    Args:
        K: The integer number of counterparties.
        uncertainity_model: The quantum circuit modelling the uncertainty model
            of counterparties' default.
        lgd: The list of losses given default of each counterparty
        target_loss: The value for which we want compute the probability such 
            that the total loss is less.
        phases: The list of phases to be used in QSP to apply the even
            threshold function.
        poly: The list of coefficients defining the even threshold function. It
            is used only id phases is null. 
            **Caution**: transformation from polynomial coefficients to phases
            uses the library pyqsp, but only old versions with low degree
            polynomials approximations work.
        threshold: the threshold value of the even threshold function we are
            interested in.
        enable_witch: deprecated.
        epsilon: deprecated
        verbose: if True, some additional information is printed.

    Returns:
        a 2-tuple of circuits. The first one is the entire circuit, whereas the
            second one is the sub-circuit to which QSVT is applied, that is the
            circuit mapping each scenario of default to its total loss.
    
    """
    offsets = lgd

    # First step: 
    # Our aim is having a state such that values corresponding to losses
    # **above** _target_loss_ are mapped to points of Bloch sphere whose
    # amplitude on the target qubit is greater than _threshold_.
    # This is achieved by mapping losses to angles such that values
    # **above** _target_loss_ correspond to angles whose sin is greater
    # then _threshold_.

    # To do so, we fix 0 as the minimum angle and pi/2 as the maximum and we
    # linearly spread values between these boundaries so that the _target_loss_
    # is mapped to _threshold_

    maximum = sum(offsets)
    target_loss_scaled = target_loss/maximum
    minimum = target_loss_scaled
    arc_threshold = np.arcsin(threshold)
    maximum_angle = np.pi/2
    minimum_angle = 0
    maximum_range = None
    minimum_range = None
   
    if target_loss_scaled > (arc_threshold - minimum_angle)/(maximum_angle - minimum_angle): # 
        # in questo caso ho più spazio per i valori esclusi dalla threshold function che prendono
        # un arco maggiore, quindi gli faccio occupare tutto lo spazio disponibile e regolo
        # quelli presi dalla threshold di conseguenza
        minimum_range = minimum_angle
        unitary_gap = (arc_threshold - minimum_range) / target_loss_scaled
    else:
        maximum_range = maximum_angle
        unitary_gap = (maximum_range - arc_threshold) / (1-target_loss_scaled)
    

    transform_losses_to_angles = lambda loss: unitary_gap*(loss/maximum - target_loss_scaled) + arc_threshold
    scaled_offsets = [transform_losses_to_angles(off) - transform_losses_to_angles(0) for off in offsets] # shifto tutto per fare in modo che lo 0 sia in 0, così che la trasformazione sia additiva
    
    if verbose:
        print(f' unitary gap {unitary_gap} ---------------------')
        print(f' minimum_angle {minimum_angle} ---------------------')
        print(f' minimum_range {minimum_range} ---------------------')
        print(f' maximum_range {maximum_range} ---------------------')
        print(f' arc_threshold {arc_threshold} ---------------------')
        print(f' scaled offset {scaled_offsets} ---------------------')
        print(f' shift {transform_losses_to_angles(0)} ---------------------')

    
    objective_e_loss = AmplitudeLoadingVar(K, scaled_offsets, starting_offset = transform_losses_to_angles(0))
    
    qsvt = qsvt_application_circuit(objective_e_loss, poly=poly, phases=phases)
    state_preparation = QuantumCircuit(
        QuantumRegister(uncertainity_model.num_qubits - K, 'z'),
        *qsvt.qregs,
        name="state_preparation"
    )
    state_preparation.append(uncertainity_model, state_preparation.qubits[0:len(state_preparation.qubits)-2])
    state_preparation.append(qsvt, qsvt.qubits[:])
    return state_preparation, objective_e_loss, True


def build_state_preparation_circuit(uncertainity_model, objective_e_loss, K) -> QuantumCircuit:
    # define the registers for convenience and readability
    qr_state = QuantumRegister(uncertainity_model.num_qubits, 'state')
    qr_obj = QuantumRegister(1, 'objective')
    ar = QuantumRegister(objective_e_loss.num_ancillas, "work")  # additional qubits

    # define the circuit
    state_preparation = QuantumCircuit(qr_state, qr_obj, ar, name='A')

    # load the random variable
    state_preparation.append(uncertainity_model.to_gate(), qr_state)

    # linear objective function (does aggregation and comparison)
        
    state_preparation.append(objective_e_loss, qr_state[-K:] + qr_obj[:] + ar[:])

    return state_preparation


def qsvt_application_circuit(p_sp: QuantumCircuit, phases=None, poly:list=None, factor=1): ### da finre
    return QSVT(
        p_sp, 
        [p_sp.num_qubits - 1], [p_sp.num_qubits - 1], # |0\rangle\langle 0|
        poly=poly,
        phases=phases, 
        adjust_conventions=True,
        ctrl_zero_qubits1=[0],
        ctrl_zero_qubits2=[0]
    )#######################
    circuit = QuantumCircuit(p_sp.qregs[0], p_sp.qregs[1], name='QSVT_'+str(factor))
    q_target = p_sp.qregs[1]
    sp_gate = p_sp.to_gate()
    sp_gate_inverse = sp_gate.inverse()
    display_=False
    for i in range(int(len(phis)/2),0,-1):
        circuit.append(sp_gate, p_sp.qregs[0][:]+p_sp.qregs[1][:])
        circuit.append(RZGate(factor*2*phis[2*i], '~ Pi_phi_'+str(2*i)), q_target) 
        circuit.append(sp_gate_inverse, p_sp.qregs[0][:]+p_sp.qregs[1][:])
        circuit.append(build_mcrz(len(p_sp.qregs[0])+len(p_sp.qregs[1]), factor*phis[2*i-1], 'Pi_phi_'+str(2*i-1)), p_sp.qregs[0][:]+p_sp.qregs[1][:]) 
        if display_:
            display_ = False
            display(circuit.decompose().draw('mpl'))
    circuit.append(sp_gate, p_sp.qregs[0][:]+p_sp.qregs[1][:])
    circuit.append(RZGate(2*phis[0], '~ Pi_phi_'+str(0)), q_target)
    return circuit

def build_mcrz(n_qubits:int, phi:float , label:str):
    ds =  ([np.exp(-phi*1j)]*(2**n_qubits -2)) + [np.exp(phi*1j), np.exp(-phi*1j)]
    U = np.diag(ds)
    # print(U)
    return UnitaryGate(U, label)

def build_rz(n_qubits:int, phi:float , label:str):
    ds = [np.exp(-phi*1j), np.exp(phi*1j)] + ([np.exp(-phi*1j)]*(2**n_qubits -2))
    U = np.diag(ds)
    return UnitaryGate(U, label)
    
