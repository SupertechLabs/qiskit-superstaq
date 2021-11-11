# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
from typing import List, Union, Optional

import qiskit
import requests

import applications_superstaq
import qiskit_superstaq as qss
from applications_superstaq import superstaq_client


class SuperstaQProvider(qiskit.providers.ProviderV1):
    """Provider for SuperstaQ backend.

    Typical usage is:

    .. code-block:: python

        import qiskit_superstaq as qss

        ss_provider = qss.superstaq_provider.SuperstaQProvider('MY_TOKEN')

        backend = ss_provider.get_backend('my_backend')

    where `'MY_TOKEN'` is the access token provided by SuperstaQ,
    and 'my_backend' is the name of the desired backend.

    Attributes:
        access_token (str): The access token.
        name (str): Name of the provider instance.
        url (str): The url that the API is hosted on.
    """

    def __init__(
        self,
        remote_host: Optional[str] = None,
        api_key: Optional[str] = None,
        default_target: str = None,
        api_version: str = applications_superstaq.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
        ibmq_token: str = None,
        ibmq_group: str = None,
        ibmq_project: str = None,
        ibmq_hub: str = None,
        ibmq_pulse: bool = True,
    ) -> None:
        self._name = "superstaq_provider"
        self.remote_host = (
            remote_host or os.getenv("SUPERSTAQ_REMOTE_HOST") or applications_superstaq.API_URL
        )
        self.api_key = api_key or os.getenv("SUPERSTAQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Parameter api_key was not specified and the environment variable "
                "SUPERSTAQ_API_KEY was also not set."
            )

        self._client = superstaq_client._SuperstaQClient(
            client_name="qiskit-superstaq",
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
            ibmq_token=ibmq_token,
            ibmq_group=ibmq_group,
            ibmq_project=ibmq_project,
            ibmq_hub=ibmq_hub,
            ibmq_pulse=ibmq_pulse,
        )

    def __str__(self) -> str:
        return f"<SuperstaQProvider(name={self._name})>"

    def __repr__(self) -> str:
        repr1 = f"<SuperstaQProvider(name={self._name}, "
        return repr1 + f"api_key={self.api_key})>"

    def get_backend(self, backend: str) -> "qss.superstaq_backend.SuperstaQBackend":
        return qss.superstaq_backend.SuperstaQBackend(provider=self, remote_host=self.remote_host, backend=backend)

    def get_client(self):
        return self._client

    def get_api_key(self) -> str:
        return self.api_key

    def backends(self) -> List[qss.superstaq_backend.SuperstaQBackend]:
        # needs to be fixed (#469)
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
                qss.superstaq_backend.SuperstaQBackend(provider=self, remote_host=self.remote_host, backend=name)
            )

        return backends

    def _http_headers(self) -> dict:
        return {
            "Authorization": self.get_api_key(),
            "Content-Type": "application/json",
            "X-Client-Name": "qiskit-superstaq",
            "X-Client-Version": qss.API_VERSION,
        }

    def aqt_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        target: str = "keysight",
    ) -> "qss.compiler_output.CompilerOutput":
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: qiskit QuantumCircuit(s)
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized qiskit.QuantumCircuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        """
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_list = not isinstance(circuits, qiskit.QuantumCircuit)

        json_dict = self._client.aqt_compile(serialized_circuits, target)

        from qiskit_superstaq import compiler_output

        return compiler_output.read_json_aqt(json_dict, circuits_list)

    def qscout_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        target: str = "qscout",
    ) -> "qss.compiler_output.CompilerOutput":
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: qiskit QuantumCircuit(s)
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized qiskit.QuantumCircuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        """
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = {"qiskit_circuits": serialized_circuits, "backend": target}
        json_dict = self._client.qscout_compile(json_dict, target)

        from qiskit_superstaq import compiler_output

        return compiler_output.read_json_qscout(json_dict, circuits_list)
