from unittest.mock import MagicMock, patch

import applications_superstaq
import qiskit

import qiskit_superstaq as qss


def test_provider() -> None:
    ss_provider = qss.superstaq_provider.SuperstaQProvider(access_token="MY_TOKEN")

    assert str(ss_provider.get_backend("ibmq_qasm_simulator")) == str(
        qss.superstaq_backend.SuperstaQBackend(
            provider=ss_provider,
            url=qss.API_URL,
            backend="ibmq_qasm_simulator",
        )
    )

    assert str(ss_provider) == "<SuperstaQProvider(name=superstaq_provider)>"

    assert (
        repr(ss_provider) == "<SuperstaQProvider(name=superstaq_provider, access_token=MY_TOKEN)>"
    )

    backend_names = [
        "aqt_device",
        "ionq_device",
        "rigetti_device",
        "ibmq_botoga",
        "ibmq_casablanca",
        "ibmq_jakarta",
        "ibmq_qasm_simulator",
    ]

    backends = []
    for name in backend_names:
        backends.append(
            qss.superstaq_backend.SuperstaQBackend(
                provider=ss_provider, url=qss.API_URL, backend=name
            )
        )

    assert ss_provider.backends() == backends


@patch("requests.post")
def test_aqt_compile(mock_post: MagicMock) -> None:
    provider = qss.superstaq_provider.SuperstaQProvider(access_token="MY_TOKEN")

    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "state_jp": applications_superstaq.converters.serialize({}),
        "pulse_lists_jp": applications_superstaq.converters.serialize([[[]]]),
    }
    out = provider.aqt_compile(qc)
    assert out.circuit == qc
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    out = provider.aqt_compile([qc])
    assert out.circuits == [qc]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "state_jp": applications_superstaq.converters.serialize({}),
        "pulse_lists_jp": applications_superstaq.converters.serialize([[[]], [[]]]),
    }
    out = provider.aqt_compile([qc, qc])
    assert out.circuits == [qc, qc]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


@patch("requests.get")
def test_backends(mock_get: MagicMock) -> None:
    provider = qss.superstaq_provider.SuperstaQProvider(access_token="MY_TOKEN")

    expected_backends = {
        "superstaq_backends": {
            "compile-and-run": [
                "ibmq_qasm_simulator",
                "ibmq_armonk_qpu",
                "ibmq_santiago_qpu",
                "ibmq_bogota_qpu",
                "ibmq_lima_qpu",
                "ibmq_belem_qpu",
                "ibmq_quito_qpu",
                "ibmq_statevector_simulator",
                "ibmq_mps_simulator",
                "ibmq_extended-stabilizer_simulator",
                "ibmq_stabilizer_simulator",
                "ibmq_manila_qpu",
                "d-wave_advantage-system1.1_qpu",
                "aws_dm1_simulator",
                "aws_tn1_simulator",
                "ionq_ion_qpu",
                "d-wave_dw-2000q-6_qpu",
                "d-wave_advantage-system4.1_qpu",
                "aws_sv1_simulator",
                "rigetti_aspen-9_qpu",
            ],
            "compile-only": ["aqt_keysight_qpu", "sandia_qscout_qpu"],
        }
    }
    mock_get.return_value.json = lambda: expected_backends
    out = provider.backends()
    assert out == expected_backends
