from qiskit.circuit.library import UGate, CXGate
import numpy as np
from qiskit import QuantumCircuit, transpile


def vector_to_tensor(psi, n):
    return psi.reshape([2] * n).transpose(list(reversed(range(n))))


def tensor_to_vector(tensor, n):
    return tensor.transpose(list(reversed(range(n)))).reshape(2**n)


def apply_u(state, u, qubit):
    n = state.ndim
    old_idx = list(range(n))
    new_idx = old_idx.copy()

    new_idx[qubit] = n

    return np.einsum(u, [n, qubit], state, old_idx, new_idx)


def apply_cx(state, cx, control, target):
    n = state.ndim
    old_idx = list(range(n))
    new_idx = old_idx.copy()

    new_idx[control] = n
    new_idx[target] = n + 1

    return np.einsum(cx, [n, n + 1, control, target], state, old_idx, new_idx)


def simulate_circuit(qc):
    n = qc.num_qubits

    # |0...0>
    psi = np.zeros(2**n)
    psi[0] = 1

    state = vector_to_tensor(psi, n)

    # vorbereiten
    cx = CXGate().to_matrix().reshape(2, 2, 2, 2)

    for instr, qargs, _ in qc.data:
        name = instr.name
        qubits = [qc.find_bit(q).index for q in qargs]

        if name == "u":
            u = UGate(*instr.params).to_matrix()
            state = apply_u(state, u, qubits[0])

        elif name == "cx":
            state = apply_cx(state, cx, qubits[0], qubits[1])

        else:
            raise ValueError(f"Gate {name} nicht unterstützt")

    return tensor_to_vector(state, n)


qc = QuantumCircuit(3)

qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)


qc = transpile(qc, basis_gates=["u", "cx"])

result = simulate_circuit(qc)
print(result)
