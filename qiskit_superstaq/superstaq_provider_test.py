import textwrap
from unittest import mock
from unittest.mock import MagicMock, patch

import applications_superstaq
import numpy as np
import pytest
import qiskit

import qubovert as qv
import qiskit_superstaq as qss


def test_provider() -> None:
    ss_provider = qss.superstaq_provider.SuperstaQProvider(api_key="MY_TOKEN")

    with pytest.raises(EnvironmentError, match="api_key was not "):
        qss.superstaq_provider.SuperstaQProvider()

    assert str(ss_provider.get_backend("ibmq_qasm_simulator")) == str(
        qss.superstaq_backend.SuperstaQBackend(
            provider=ss_provider,
            remote_host=qss.API_URL,
            backend="ibmq_qasm_simulator",
        )
    )

    assert str(ss_provider) == "<SuperstaQProvider(name=superstaq_provider)>"

    assert repr(ss_provider) == "<SuperstaQProvider(name=superstaq_provider, api_key=MY_TOKEN)>"

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
                provider=ss_provider, remote_host=qss.API_URL, backend=name
            )
        )

    assert ss_provider.backends() == backends


@patch("requests.post")
def test_aqt_compile(mock_post: MagicMock) -> None:
    provider = qss.superstaq_provider.SuperstaQProvider(api_key="MY_TOKEN")

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


@patch("requests.post")
def test_qscout_compile(mock_post: MagicMock) -> None:
    provider = qss.superstaq_provider.SuperstaQProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)

    jaqal_program = textwrap.dedent(
        """\
                register allqubits[1]

                prepare_all
                R allqubits[0] -1.5707963267948966 1.5707963267948966
                Rz allqubits[0] -3.141592653589793
                measure_all
                """
    )

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "jaqal_programs": [jaqal_program],
    }
    out = provider.qscout_compile(qc)
    assert out.circuit == qc

    out = provider.qscout_compile([qc])
    assert out.circuits == [qc]

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = provider.qscout_compile([qc, qc])
    assert out.circuits == [qc, qc]


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.submit_qubo",
    return_value={
        "solution": applications_superstaq.converters.serialize(
            np.rec.array(
                [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
                dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
            )
        )
    },
)
def test_provider_submit_qubo(mock_submit_qubo: mock.MagicMock) -> None:
    service = qss.superstaq_provider.SuperstaQProvider(api_key="MY_TOKEN")
    expected = np.rec.array(
        [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
        dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
    )
    assert repr(service.submit_qubo(qv.QUBO(), "target", repetitions=10)) == repr(expected)


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.find_min_vol_portfolio",
    return_value={
        "best_portfolio": ["AAPL", "GOOG"],
        "best_ret": 8.1,
        "best_std_dev": 10.5,
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_find_min_vol_portfolio(mock_find_min_vol_portfolio: mock.MagicMock) -> None:
    service = qss.superstaq_provider.SuperstaQProvider(api_key="MY_TOKEN")
    qubo = {("0",): 123}
    expected = applications_superstaq.finance.MinVolOutput(["AAPL", "GOOG"], 8.1, 10.5, qubo)
    assert service.find_min_vol_portfolio(["AAPL", "GOOG", "IEF", "MMM"], 8) == expected


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.find_max_pseudo_sharpe_ratio",
    return_value={
        "best_portfolio": ["AAPL", "GOOG"],
        "best_ret": 8.1,
        "best_std_dev": 10.5,
        "best_sharpe_ratio": 0.771,
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_find_max_pseudo_sharpe_ratio(
    mock_find_max_pseudo_sharpe_ratio: mock.MagicMock,
) -> None:
    service = qss.superstaq_provider.SuperstaQProvider(api_key="MY_TOKEN")
    qubo = {("0",): 123}
    expected = applications_superstaq.finance.MaxSharpeOutput(
        ["AAPL", "GOOG"], 8.1, 10.5, 0.771, qubo
    )
    assert service.find_max_pseudo_sharpe_ratio(["AAPL", "GOOG", "IEF", "MMM"], k=0.5) == expected
