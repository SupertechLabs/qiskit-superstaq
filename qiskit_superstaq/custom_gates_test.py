from typing import Set

import numpy as np
import pytest
import qiskit

import qiskit_superstaq


def _check_gate_definition(gate: qiskit.circuit.Gate) -> None:
    """Check gate.definition, gate.__array__(), and gate.inverse() against one another"""

    assert np.allclose(gate.to_matrix(), gate.__array__())
    defined_operation = qiskit.quantum_info.Operator(gate.definition)
    assert defined_operation.is_unitary()
    assert defined_operation.equiv(gate.to_matrix(), atol=1e-10)

    inverse_operation = qiskit.quantum_info.Operator(gate.inverse().definition)
    assert inverse_operation.is_unitary()

    assert inverse_operation.equiv(gate.inverse().to_matrix(), atol=1e-10)
    assert inverse_operation.equiv(gate.to_matrix().T.conj(), atol=1e-10)


def test_acecr() -> None:
    gate = qiskit_superstaq.AceCR("+-")
    _check_gate_definition(gate)
    assert repr(gate) == "qiskit_superstaq.AceCR('+-')"
    assert str(gate) == "AceCR+-"
    assert gate.qasm() == "acecr_pm"

    gate = qiskit_superstaq.AceCR("-+", label="label")
    _check_gate_definition(gate)
    assert repr(gate) == "qiskit_superstaq.AceCR('-+', label='label')"
    assert str(gate) == "AceCR-+"
    assert gate.qasm() == "acecr_mp"

    gate = qiskit_superstaq.AceCR("-+", sandwich_rx_rads=np.pi / 2)
    _check_gate_definition(gate)
    assert repr(gate) == "qiskit_superstaq.AceCR('-+', sandwich_rx_rads=1.5707963267948966)"
    assert str(gate) == "AceCR-+|RXGate(pi/2)|"
    assert gate.qasm() == "acecr_mp_rx(pi/2)"

    gate = qiskit_superstaq.AceCR("-+", sandwich_rx_rads=np.pi / 2, label="label")
    _check_gate_definition(gate)
    assert (
        repr(gate)
        == "qiskit_superstaq.AceCR('-+', sandwich_rx_rads=1.5707963267948966, label='label')"
    )
    assert str(gate) == "AceCR-+|RXGate(pi/2)|"
    assert gate.qasm() == "acecr_mp_rx(pi/2)"

    with pytest.raises(ValueError, match="Polarity must be"):
        _ = qiskit_superstaq.AceCR("++")


def test_zz_swap() -> None:
    gate = qiskit_superstaq.ZZSwapGate(1.23)
    _check_gate_definition(gate)
    assert repr(gate) == "qiskit_superstaq.ZZSwapGate(1.23)"
    assert str(gate) == "ZZSwapGate(1.23)"

    gate = qiskit_superstaq.ZZSwapGate(4.56, label="label")
    assert repr(gate) == "qiskit_superstaq.ZZSwapGate(4.56, label='label')"
    assert str(gate) == "ZZSwapGate(4.56)"


def test_parallel_gates() -> None:
    gate = qiskit_superstaq.ParallelGates(
        qiskit_superstaq.AceCR("+-"),
        qiskit.circuit.library.RXGate(1.23),
    )
    assert str(gate) == "ParallelGates(acecr_pm, rx(1.23))"
    _check_gate_definition(gate)

    # confirm gates are applied to disjoint qubits
    all_qargs: Set[qiskit.circuit.Qubit] = set()
    for _, qargs, _ in gate.definition:
        assert all_qargs.isdisjoint(qargs)
        all_qargs.update(qargs)
    assert len(all_qargs) == gate.num_qubits

    # double check qubit ordering
    qc1 = qiskit.QuantumCircuit(3)
    qc1.append(gate, [0, 2, 1])

    qc2 = qiskit.QuantumCircuit(3)
    qc2.rx(1.23, 1)
    qc2.append(qiskit_superstaq.AceCR("+-"), [0, 2])

    assert qiskit.quantum_info.Operator(qc1).equiv(qc2, atol=1e-14)

    gate = qiskit_superstaq.ParallelGates(
        qiskit.circuit.library.XGate(),
        qiskit_superstaq.ZZSwapGate(1.23),
        qiskit.circuit.library.ZGate(),
        label="label",
    )
    assert str(gate) == "ParallelGates(x, zzswap(1.23), z)"
    _check_gate_definition(gate)

    # confirm gates are applied to disjoint qubits
    all_qargs.clear()
    for _, qargs, _ in gate.definition:
        assert all_qargs.isdisjoint(qargs)
        all_qargs.update(qargs)
    assert len(all_qargs) == gate.num_qubits

    gate = qiskit_superstaq.ParallelGates(
        qiskit.circuit.library.XGate(),
        qiskit_superstaq.ParallelGates(
            qiskit.circuit.library.YGate(),
            qiskit.circuit.library.ZGate(),
        ),
    )
    gate2 = qiskit_superstaq.ParallelGates(
        qiskit.circuit.library.XGate(),
        qiskit.circuit.library.YGate(),
        qiskit.circuit.library.ZGate(),
    )
    assert gate.component_gates == gate2.component_gates
    print(gate.definition)
    print(gate2.definition)
    assert gate == gate2

    with pytest.raises(ValueError, match="Component gates must be"):
        _ = qiskit_superstaq.ParallelGates(qiskit.circuit.Measure())


def test_aqticcx() -> None:
    gate = qiskit_superstaq.AQTiCCXGate()
    _check_gate_definition(gate)

    assert repr(gate) == "qiskit_superstaq.iCCXGate(label=None, ctrl_state=0)"
    assert str(gate) == "iCCXGate(label=None, ctrl_state=0)"

    qc = qiskit.QuantumCircuit(3)

    qc.append(qiskit_superstaq.AQTiCCXGate(), [0, 1, 2])

    correct_unitary = np.array(
        [
            [0, 0, 0, 0, 1j, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1j, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
    )

    np.allclose(qiskit.quantum_info.Operator(qc), correct_unitary)


def test_iccxdg() -> None:
    gate = qiskit_superstaq.custom_gates.iCCXdgGate()
    _check_gate_definition(gate)
    assert repr(gate) == "qiskit_superstaq.iCCXdgGate(label=None, ctrl_state=3)"
    assert str(gate) == "iCCXdgGate(label=None, ctrl_state=3)"


def test_custom_resolver() -> None:
    custom_gates = [
        qiskit_superstaq.AceCR("+-"),
        qiskit_superstaq.ZZSwapGate(1.23),
        qiskit_superstaq.AQTiCCXGate(),
        qiskit_superstaq.custom_gates.iCCXGate(),
        qiskit_superstaq.custom_gates.iCCXGate(ctrl_state="01"),
        qiskit_superstaq.custom_gates.iCCXGate(ctrl_state="10"),
    ]

    generic_gates = [
        qiskit.circuit.Gate(gate.name, gate.num_qubits, gate.params) for gate in custom_gates
    ]

    for custom_gate, generic_gate in zip(custom_gates, generic_gates):
        resolved_gate = qiskit_superstaq.custom_gates.custom_resolver(generic_gate)
        assert resolved_gate == custom_gate
        assert qiskit_superstaq.custom_gates.custom_resolver(custom_gate) is custom_gate

    parallel_gates = qiskit_superstaq.ParallelGates(
        qiskit.circuit.library.RXGate(4.56), qiskit.circuit.library.CXGate(), *custom_gates
    )
    parallel_generic_gates = qiskit_superstaq.ParallelGates(
        qiskit.circuit.library.RXGate(4.56), qiskit.circuit.library.CXGate(), *generic_gates
    )
    generic_parallel_gates = parallel_generic_gates.definition.to_gate()
    generic_parallel_gates.name = parallel_gates.name

    assert qiskit_superstaq.custom_gates.custom_resolver(parallel_generic_gates) == parallel_gates
    assert qiskit_superstaq.custom_gates.custom_resolver(generic_parallel_gates) == parallel_gates

    for gate in (
        qiskit.circuit.library.CXGate(),
        qiskit.circuit.library.RXGate(1.23),
        qiskit.circuit.Gate("unknown", 1, []),
    ):
        assert qiskit_superstaq.custom_gates.custom_resolver(gate) is gate
