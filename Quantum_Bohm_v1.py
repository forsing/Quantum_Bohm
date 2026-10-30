from __future__ import annotations

# Quantum_Bohm

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit
import scipy

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HamiltonianGate
from qiskit.quantum_info import Operator, Pauli, Statevector
from scipy.linalg import expm_frechet
from scipy.optimize import minimize
from scipy.special import logsumexp


FILES = (
    Path("/data/loto7_4710_k86.csv"),
    Path("/data/loto7_4710_k86_loto_2977.csv"),
    Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBERS = 39
PICKS = 7
QUBITS = 6
DIMENSION = 1 << QUBITS
PARAMETERS = 30

TIME_STEP = 0.35
HAMILTONIAN_SCALE = 0.20
INPUT_PHASE = 0.45
PARAMETER_LIMIT = 2.0

RIDGES = (0.03, 0.3)
MAX_ITERATIONS = 120
MAX_FUNCTION_CALLS = 1000
CHECKPOINT_EVERY = 5
GRADIENT_LIMIT = 5.0e-5

VERSION = "QUANTUM_BOHM_V1_20261029"

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CHECKPOINT_DIRECTORY = (
    SCRIPT_DIRECTORY / "Quantum_Bohm_V1_checkpoints"
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(values, label):
    require(
        np.all(np.isfinite(values)),
        f"{label}: nedefinisana ili nekonačna vrednost.",
    )


def pauli_label(operators):
    symbols = ["I"] * QUBITS

    for qubit, symbol in operators:
        symbols[QUBITS - 1 - qubit] = symbol

    return "".join(symbols)


class BohmMemory:
    def __init__(self):
        labels = []

        for axis in ("X", "Y", "Z"):
            for qubit in range(QUBITS):
                labels.append(
                    pauli_label(((qubit, axis),))
                )

        for axis in ("Z", "X"):
            for qubit in range(QUBITS):
                other = (qubit + 1) % QUBITS

                labels.append(
                    pauli_label(
                        (
                            (qubit, axis),
                            (other, axis),
                        )
                    )
                )

        require(
            len(labels) == PARAMETERS
            and len(set(labels)) == PARAMETERS,
            "Neispravna baza Hamiltonijana.",
        )

        self.labels = tuple(labels)

        self.basis = np.asarray(
            [
                Pauli(label).to_matrix()
                for label in self.labels
            ],
            dtype=np.complex128,
        )

        self.base_coefficients = np.asarray(
            [
                *[
                    0.32 + 0.03 * qubit
                    for qubit in range(QUBITS)
                ],
                *[
                    0.08 * math.cos(qubit + 0.4)
                    for qubit in range(QUBITS)
                ],
                *[
                    0.10 * math.sin(qubit + 0.7)
                    for qubit in range(QUBITS)
                ],
                *[
                    0.09 * math.cos(qubit + 0.2)
                    for qubit in range(QUBITS)
                ],
                *[
                    0.07 * math.sin(qubit + 0.3)
                    for qubit in range(QUBITS)
                ],
            ],
            dtype=float,
        )

        preparation = QuantumCircuit(QUBITS)

        for qubit in range(QUBITS):
            preparation.ry(
                1.03 + 0.06 * qubit,
                qubit,
            )

            preparation.rz(
                0.13 * (qubit + 1),
                qubit,
            )

        for qubit in range(QUBITS - 1):
            preparation.cx(qubit, qubit + 1)

        self.preparation = preparation

        self.initial_state = np.asarray(
            Statevector.from_instruction(
                preparation
            ).data,
            dtype=np.complex128,
        )

        require(
            abs(
                float(
                    np.sum(
                        np.abs(self.initial_state) ** 2
                    )
                ) - 1.0
            ) < 1.0e-11,
            "Početno stanje nije normalizovano.",
        )

    @staticmethod
    def initial_parameters():
        return np.zeros(PARAMETERS, dtype=float)

    @staticmethod
    def input_factors(occupancy):
        occupancy = np.asarray(occupancy, dtype=float)

        require(
            occupancy.ndim == 2
            and occupancy.shape[1] == NUMBERS,
            "Pogrešan oblik istorijskih kombinacija.",
        )

        require(
            np.all(
                (occupancy == 0.0)
                | (occupancy == 1.0)
            )
            and np.all(
                np.sum(occupancy, axis=1) == PICKS
            ),
            "Istorijski ulaz nije legalna kombinacija.",
        )

        phases = np.zeros(
            (len(occupancy), DIMENSION),
            dtype=float,
        )

        phases[:, :NUMBERS] = (
            INPUT_PHASE * occupancy
        )

        return np.exp(1j * phases)

    def hamiltonian(self, parameters):
        parameters = np.asarray(parameters, dtype=float)

        require(
            parameters.shape == (PARAMETERS,),
            "Pogrešan broj parametara Hamiltonijana.",
        )

        finite(parameters, "Parametri")

        coefficients = (
            self.base_coefficients
            + HAMILTONIAN_SCALE * parameters
        )

        matrix = np.einsum(
            "p,pij->ij",
            coefficients,
            self.basis,
            optimize=True,
        )

        finite(matrix, "Hamiltonijan")

        require(
            np.max(
                np.abs(matrix - matrix.conj().T)
            ) < 1.0e-12,
            "Hamiltonijan nije Hermitski.",
        )

        return matrix

    def evolution(self, parameters):
        hamiltonian = self.hamiltonian(parameters)

        gate = HamiltonianGate(
            hamiltonian,
            time=TIME_STEP,
            label="Learned_H",
        )

        circuit = QuantumCircuit(QUBITS)
        circuit.append(gate, list(range(QUBITS)))

        unitary = np.asarray(
            Operator(circuit).data,
            dtype=np.complex128,
        )

        finite(unitary, "Qiskit evolucija")

        error = np.max(
            np.abs(
                unitary.conj().T @ unitary
                - np.eye(DIMENSION)
            )
        )

        require(
            error < 2.0e-10,
            "Qiskit evolucija nije numerički unitarna.",
        )

        return hamiltonian, unitary

    def forward(
        self,
        parameters,
        factors,
        reset_memory=False,
    ):
        factors = np.asarray(
            factors,
            dtype=np.complex128,
        )

        require(
            factors.ndim == 2
            and factors.shape[1] == DIMENSION
            and len(factors) > 0,
            "Pogrešan oblik ulaznih faza.",
        )

        require(
            np.max(
                np.abs(np.abs(factors) - 1.0)
            ) < 1.0e-12,
            "Ulazni faktori nisu čiste faze.",
        )

        hamiltonian, unitary = self.evolution(parameters)

        states = np.empty(
            (len(factors) + 1, DIMENSION),
            dtype=np.complex128,
        )

        states[0] = self.initial_state

        for index, factor in enumerate(factors):
            previous = (
                self.initial_state
                if reset_memory
                else states[index]
            )

            before_evolution = previous * factor

            states[index + 1] = (
                unitary @ before_evolution
            )

        finite(states, "Kvantna memorija")

        norm_error = np.max(
            np.abs(
                np.sum(
                    np.abs(states) ** 2,
                    axis=1,
                ) - 1.0
            )
        )

        require(
            norm_error < 2.0e-8,
            "Akumulirana greška norme kvantne memorije je prevelika.",
        )

        weights = np.abs(
            states[1:, :NUMBERS]
        ) ** 2

        require(
            np.all(weights > 0.0),
            "Kandidat ima nultu težinu u logaritamskom domenu.",
        )

        require(
            np.all(
                np.sum(weights, axis=1)
                <= 1.0 + 2.0e-8
            ),
            "Masa brojeva 1–39 prelazi jedan.",
        )

        return weights, states, hamiltonian, unitary

    def verify_qiskit_history(
        self,
        parameters,
        factors,
        expected_state,
    ):
        _, unitary = self.evolution(parameters)
        evolution_operator = Operator(unitary)

        reference = Statevector(
            self.initial_state.copy()
        )

        for factor in factors:
            phase_operator = Operator(
                np.diag(factor)
            )

            reference = reference.evolve(
                phase_operator
            )

            reference = reference.evolve(
                evolution_operator
            )

        error = float(
            np.max(
                np.abs(
                    reference.data - expected_state
                )
            )
        )

        require(
            error < 2.0e-8,
            "Puna istorija odstupa od Qiskit Statevector evolucije.",
        )

        return error

    def seven_register_step(self, parameters, factor):
        registers = [
            QuantumRegister(QUBITS, f"r{index + 1}")
            for index in range(PICKS)
        ]

        circuit = QuantumCircuit(*registers)

        hamiltonian = self.hamiltonian(parameters)

        for register in registers:
            circuit.unitary(
                np.diag(factor),
                list(register),
                label="Observed_draw",
            )

            circuit.append(
                HamiltonianGate(
                    hamiltonian,
                    time=TIME_STEP,
                    label="Learned_H",
                ),
                list(register),
            )

        require(
            circuit.num_qubits == PICKS * QUBITS,
            "Neispravan broj kubita u sedam registara.",
        )

        return circuit


def subset_distribution(
    weights,
    picks=PICKS,
    marginals=False,
):
    weights = np.asarray(weights, dtype=float)

    require(
        weights.ndim == 2 and len(weights) > 0,
        "Neispravan oblik težina.",
    )

    samples, count = weights.shape

    require(
        0 <= picks <= count,
        "Neispravan broj izabranih brojeva.",
    )

    require(
        np.all(np.isfinite(weights))
        and np.all(weights > 0.0),
        "Težine moraju biti konačne i pozitivne.",
    )

    log_weights = np.log(weights)

    forward = np.full(
        (samples, count + 1, picks + 1),
        -np.inf,
        dtype=float,
    )

    forward[:, 0, 0] = 0.0

    for index in range(count):
        forward[:, index + 1] = forward[:, index]

        if picks > 0:
            forward[:, index + 1, 1:] = np.logaddexp(
                forward[:, index, 1:],
                log_weights[:, index, None]
                + forward[:, index, :picks],
            )

    log_z = forward[:, count, picks]
    finite(log_z, "Normalizacija podskupova")

    if not marginals:
        return log_z, None

    if picks == 0:
        return log_z, np.zeros_like(weights)

    backward = np.full_like(forward, -np.inf)
    backward[:, count, 0] = 0.0

    for index in range(count - 1, -1, -1):
        backward[:, index] = backward[:, index + 1]

        backward[:, index, 1:] = np.logaddexp(
            backward[:, index + 1, 1:],
            log_weights[:, index, None]
            + backward[:, index + 1, :picks],
        )

    require(
        np.max(
            np.abs(
                backward[:, 0, picks] - log_z
            )
        ) < 2.0e-10,
        "Dve logaritamske normalizacije se ne slažu.",
    )

    inclusion = np.empty_like(weights)

    for index in range(count):
        rest = logsumexp(
            forward[:, index, :picks]
            + backward[
                :,
                index + 1,
                picks - 1::-1,
            ],
            axis=1,
        )

        inclusion[:, index] = np.exp(
            log_weights[:, index]
            + rest
            - log_z
        )

    finite(inclusion, "Marginalne verovatnoće")

    require(
        np.all(inclusion >= 0.0)
        and np.all(inclusion <= 1.0 + 2.0e-8),
        "Marginalna verovatnoća je izvan granica.",
    )

    require(
        np.max(
            np.abs(
                np.sum(inclusion, axis=1) - picks
            )
        ) < 2.0e-7,
        "Zbir marginala nije jednak broju izbora.",
    )

    return log_z, inclusion


def subset_log_probability(weights, targets):
    require(
        targets.shape == weights.shape,
        "Ciljevi i težine nisu istog oblika.",
    )

    log_z, _ = subset_distribution(weights)

    result = (
        np.sum(
            targets * np.log(weights),
            axis=1,
        ) - log_z
    )

    finite(result, "Logaritamske verovatnoće")

    require(
        np.all(result <= 2.0e-8),
        "Verovatnoća sedmorke prelazi jedan.",
    )

    return result


def exact_maximum(weights, picks=PICKS):
    weights = np.asarray(weights, dtype=float)

    require(
        weights.ndim == 1
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0),
        "Neispravne težine za maksimum.",
    )

    require(
        0 <= picks <= len(weights),
        "Neispravna veličina maksimuma.",
    )

    order = np.argsort(
        -weights,
        kind="stable",
    )

    selected = tuple(
        int(value)
        for value in np.sort(order[:picks])
    )

    log_z, _ = subset_distribution(
        weights[None, :],
        picks=picks,
    )

    log_probability = (
        float(
            np.sum(
                np.log(weights[list(selected)])
            )
        ) - float(log_z[0])
    )

    require(
        -math.log(
            math.comb(len(weights), picks)
        ) - 2.0e-8
        <= log_probability
        <= 2.0e-8,
        "Maksimum ne prolazi granice verovatnoće.",
    )

    return selected, log_probability


def alternative_log_normalizer(weights, picks=PICKS):
    weights = np.asarray(
        weights,
        dtype=np.longdouble,
    )

    scale = np.max(weights)
    require(scale > 0.0, "Nulta skala normalizacije.")

    coefficients = np.zeros(
        picks + 1,
        dtype=np.longdouble,
    )

    coefficients[0] = 1.0

    for index, weight in enumerate(weights / scale):
        for degree in range(
            min(picks, index + 1),
            0,
            -1,
        ):
            coefficients[degree] += (
                weight * coefficients[degree - 1]
            )

    require(
        coefficients[picks] > 0.0,
        "Alternativna normalizacija nije pozitivna.",
    )

    return float(
        np.log(coefficients[picks])
        + picks * np.log(scale)
    )


def objective(
    parameters,
    source,
    factors,
    targets,
    ridge,
):
    weights, states, hamiltonian, unitary = source.forward(
        parameters,
        factors,
    )

    require(
        len(weights) == len(targets),
        "Broj ulaza i ciljeva se ne slaže.",
    )

    log_z, inclusion = subset_distribution(
        weights,
        marginals=True,
    )

    log_probability = (
        np.sum(
            targets * np.log(weights),
            axis=1,
        ) - log_z
    )

    nll = -float(np.mean(log_probability))

    parameters = np.asarray(parameters, dtype=float)

    value = (
        nll
        + 0.5 * ridge
        * float(np.mean(parameters ** 2))
    )

    residual = (
        inclusion - targets
    ) / len(targets)

    direct = np.zeros(
        (len(targets), DIMENSION),
        dtype=np.complex128,
    )

    direct[:, :NUMBERS] = (
        2.0
        * states[1:, :NUMBERS]
        * (residual / weights)
    )

    # Tačan povratni prolaz kroz celu istoriju.
    # Nema odsecanja gradijenta na kratak vremenski prozor.
    future_adjoint = np.zeros(
        DIMENSION,
        dtype=np.complex128,
    )

    unitary_adjoint = unitary.conj().T

    operator_gradient = np.zeros(
        (DIMENSION, DIMENSION),
        dtype=np.complex128,
    )

    for index in range(len(targets) - 1, -1, -1):
        adjoint = (
            direct[index] + future_adjoint
        )

        before_evolution = (
            states[index] * factors[index]
        )

        operator_gradient += np.outer(
            adjoint,
            before_evolution.conj(),
        )

        future_adjoint = (
            factors[index].conj()
            * (unitary_adjoint @ adjoint)
        )

    generator = -1j * TIME_STEP * hamiltonian

    exponential_adjoint, generator_gradient = expm_frechet(
        generator.conj().T,
        operator_gradient,
        compute_expm=True,
    )

    require(
        np.max(
            np.abs(
                exponential_adjoint
                - unitary_adjoint
            )
        ) < 2.0e-9,
        "Gradijent eksponencijala odstupa od Qiskit evolucije.",
    )

    directions = (
        -1j
        * TIME_STEP
        * HAMILTONIAN_SCALE
        * source.basis
    )

    gradient = np.real(
        np.einsum(
            "ij,pij->p",
            generator_gradient.conj(),
            directions,
            optimize=True,
        )
    )

    gradient += ridge * parameters / PARAMETERS

    require(
        np.isfinite(value),
        "Funkcija cilja nije konačna.",
    )

    finite(gradient, "Gradijent Hamiltonijana")

    return float(value), gradient


def projected_gradient_norm(parameters, gradient):
    projected = np.array(
        gradient,
        dtype=float,
        copy=True,
    )

    lower = (
        parameters <= -PARAMETER_LIMIT + 1.0e-10
    )

    upper = (
        parameters >= PARAMETER_LIMIT - 1.0e-10
    )

    projected[
        lower & (projected > 0.0)
    ] = 0.0

    projected[
        upper & (projected < 0.0)
    ] = 0.0

    return float(
        np.max(np.abs(projected))
    )


def current_and_guidance(hamiltonian, state):
    """
    Koristi se hbar = 1.

    J[i,j] je struja iz stanja j u stanje i.

    rate[i,j] = max(J[i,j], 0) / rho[j].

    Stope definišu diskretni zakon transporta raspodele.
    Ne izvlače se slučajni skokovi ni putanje.
    """

    state = np.asarray(
        state,
        dtype=np.complex128,
    )

    density = np.abs(state) ** 2

    current = (
        2.0
        * np.imag(
            state.conj()[:, None]
            * hamiltonian
            * state[None, :]
        )
    )

    finite(current, "Struje verovatnoće")

    require(
        np.max(
            np.abs(current + current.T)
        ) < 2.0e-10,
        "Struje nisu antisimetrične.",
    )

    rates = np.zeros_like(current)

    np.divide(
        np.maximum(current, 0.0),
        density[None, :],
        out=rates,
        where=density[None, :] > 0.0,
    )

    finite(rates, "Stope diskretnog vođenja")

    require(
        np.all(rates >= 0.0),
        "Pronađena je negativna stopa transporta.",
    )

    guided_derivative = (
        rates @ density
        - density * np.sum(rates, axis=0)
    )

    schrodinger_derivative = (
        2.0
        * np.imag(
            state.conj()
            * (hamiltonian @ state)
        )
    )

    current_derivative = np.sum(
        current,
        axis=1,
    )

    guidance_error = float(
        np.max(
            np.abs(
                guided_derivative
                - schrodinger_derivative
            )
        )
    )

    require(
        guidance_error < 2.0e-9,
        "Zakon vođenja ne čuva kvantnu raspodelu.",
    )

    require(
        np.max(
            np.abs(
                current_derivative
                - schrodinger_derivative
            )
        ) < 2.0e-10,
        "Struje ne zadovoljavaju jednačinu kontinuiteta.",
    )

    require(
        abs(
            float(
                np.sum(
                    schrodinger_derivative
                )
            )
        ) < 2.0e-10,
        "Struje ne čuvaju ukupnu verovatnoću.",
    )

    return current, rates, guidance_error


def verify_current_transport(
    hamiltonian,
    before_evolution,
    expected_after,
):
    eigenvalues, eigenvectors = np.linalg.eigh(
        hamiltonian
    )

    coefficients = (
        eigenvectors.conj().T
        @ before_evolution
    )

    initial_density = (
        np.abs(before_evolution) ** 2
    )

    expected_density = (
        np.abs(expected_after) ** 2
    )

    current_error = 0.0

    for state in (
        before_evolution,
        expected_after,
    ):
        _, _, error = current_and_guidance(
            hamiltonian,
            state,
        )

        current_error = max(
            current_error,
            error,
        )

    density_error = math.inf
    mass_error = math.inf
    used_order = None

    for order in (16, 32, 64, 128):
        nodes, quadrature_weights = (
            np.polynomial.legendre.leggauss(order)
        )

        times = (
            0.5 * TIME_STEP * (nodes + 1.0)
        )

        waves = (
            np.exp(
                -1j
                * times[:, None]
                * eigenvalues[None, :]
            )
            * coefficients[None, :]
        ) @ eigenvectors.T

        derivatives = (
            2.0
            * np.imag(
                waves.conj()
                * (waves @ hamiltonian.T)
            )
        )

        transported_density = (
            initial_density
            + 0.5
            * TIME_STEP
            * (
                quadrature_weights
                @ derivatives
            )
        )

        density_error = float(
            np.max(
                np.abs(
                    transported_density
                    - expected_density
                )
            )
        )

        mass_error = abs(
            float(
                np.sum(transported_density)
                - np.sum(initial_density)
            )
        )

        if (
            density_error < 2.0e-9
            and mass_error < 2.0e-9
            and np.min(transported_density) > -2.0e-9
        ):
            used_order = order
            break

    require(
        used_order is not None,
        "Integrisani transport struja odstupa od kvantne evolucije.",
    )

    return {
        "guidance_error": current_error,
        "transport_error": density_error,
        "transport_mass_error": mass_error,
        "quadrature_order": used_order,
    }


def load_draws(path):
    fingerprint = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    rows = []
    empty = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, row in enumerate(
            csv.reader(handle),
            start=1,
        ):
            if not row or all(
                not field.strip()
                for field in row
            ):
                empty += 1
                continue

            require(
                len(row) == PICKS,
                f"{path.name}, red {line_number}: očekujem sedam kolona.",
            )

            try:
                numbers = [
                    int(field.strip())
                    for field in row
                ]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(numbers)) == PICKS,
                f"{path.name}, red {line_number}: ponovljeni brojevi.",
            )

            require(
                all(
                    1 <= number <= NUMBERS
                    for number in numbers
                ),
                f"{path.name}, red {line_number}: broj izvan 1–39.",
            )

            rows.append(
                sorted(
                    number - 1
                    for number in numbers
                )
            )

    require(
        len(rows) >= 100,
        f"{path.name}: premalo redova.",
    )

    draws = np.asarray(
        rows,
        dtype=np.int64,
    )

    occupancy = np.zeros(
        (len(draws), NUMBERS),
        dtype=float,
    )

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {empty}",
        f"SHA256: {fingerprint}",
        "CSV redosled ostaje od najstarijeg do najnovijeg kola.",
    )

    return draws, occupancy


def checkpoint_path(
    source,
    factors,
    targets,
    ridge,
    initial,
):
    digest = hashlib.sha256()

    configuration = (
        VERSION,
        qiskit.__version__,
        np.__version__,
        scipy.__version__,
        TIME_STEP,
        HAMILTONIAN_SCALE,
        INPUT_PHASE,
        PARAMETER_LIMIT,
        MAX_ITERATIONS,
        MAX_FUNCTION_CALLS,
        float(ridge),
    )

    digest.update(
        repr(configuration).encode("utf-8")
    )

    for values in (
        factors,
        targets,
        initial,
        source.initial_state,
        source.basis,
        source.base_coefficients,
    ):
        array = np.ascontiguousarray(values)

        digest.update(
            str(array.shape).encode("ascii")
        )

        digest.update(
            str(array.dtype).encode("ascii")
        )

        digest.update(array.tobytes())

    return (
        CHECKPOINT_DIRECTORY
        / f"{digest.hexdigest()}.npz"
    )


def save_checkpoint(
    path,
    parameters,
    iterations,
    done,
    message,
):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.asarray(VERSION),
            parameters=np.asarray(
                parameters,
                dtype=float,
            ),
            iterations=np.asarray(
                iterations,
                dtype=np.int64,
            ),
            done=np.asarray(bool(done)),
            message=np.asarray(str(message)),
        )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def read_checkpoint(path):
    if not path.exists():
        return None

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as saved:
            require(
                str(
                    saved["version"].item()
                ) == VERSION,
                "Nepodudarna verzija kontrolne tačke.",
            )

            parameters = np.asarray(
                saved["parameters"],
                dtype=float,
            ).copy()

            iterations = int(
                saved["iterations"].item()
            )

            done = bool(
                saved["done"].item()
            )

            message = str(
                saved["message"].item()
            )

        require(
            parameters.shape == (PARAMETERS,)
            and np.all(np.isfinite(parameters))
            and np.all(
                np.abs(parameters)
                <= PARAMETER_LIMIT + 1.0e-10
            ),
            "Neispravni sačuvani parametri.",
        )

        require(
            0 <= iterations <= MAX_ITERATIONS,
            "Neispravan broj sačuvanih iteracija.",
        )

        return parameters, iterations, done, message

    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        BadZipFile,
        EOFError,
    ) as error:
        say(
            f"Neupotrebljiva kontrolna tačka: {path.name}",
            str(error),
        )

        return None


@dataclass
class Fitted:
    parameters: np.ndarray
    loss: float
    gradient_norm: float
    converged: bool
    iterations: int
    message: str


def fit_model(
    source,
    factors,
    targets,
    ridge,
    label,
    initial=None,
):
    require(
        len(factors) == len(targets)
        and len(targets) > 0,
        f"{label}: neispravan trening skup.",
    )

    if initial is None:
        initial = source.initial_parameters()

    initial = np.asarray(
        initial,
        dtype=float,
    ).copy()

    path = checkpoint_path(
        source,
        factors,
        targets,
        ridge,
        initial,
    )

    saved = read_checkpoint(path)

    parameters = initial.copy()
    used = 0
    done = False
    message = "Početni kandidat"

    if saved is not None:
        parameters, used, done, message = saved

        try:
            objective(
                parameters,
                source,
                factors,
                targets,
                ridge,
            )
        except (
            RuntimeError,
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            say(
                f"{label}: sačuvani kandidat nije prošao proveru.",
                "Koristim početni kandidat ovog treninga.",
            )

            parameters = initial.copy()
            used = 0
            done = False
            message = "Početak bez neispravne kontrolne tačke"

    value, gradient = objective(
        parameters,
        source,
        factors,
        targets,
        ridge,
    )

    best = {
        "parameters": parameters.copy(),
        "value": float(value),
        "gradient": gradient.copy(),
    }

    prior_iterations = used
    local_iterations = 0
    started = time.monotonic()

    say(
        "",
        label,
        f"Trening prelaza: {len(targets)} | ridge: {ridge}",
    )

    if done:
        say("Koristim završenu kontrolnu tačku.")
    else:
        save_checkpoint(
            path,
            best["parameters"],
            used,
            False,
            message,
        )

    def fun(candidate):
        candidate_value, candidate_gradient = objective(
            candidate,
            source,
            factors,
            targets,
            ridge,
        )

        if candidate_value < best["value"]:
            best["parameters"] = np.asarray(
                candidate,
                dtype=float,
            ).copy()

            best["value"] = float(candidate_value)
            best["gradient"] = candidate_gradient.copy()

        return candidate_value, candidate_gradient

    def callback(_candidate):
        nonlocal local_iterations, used

        local_iterations += 1

        used = min(
            MAX_ITERATIONS,
            prior_iterations + local_iterations,
        )

        if local_iterations % CHECKPOINT_EVERY == 0:
            gradient_norm = projected_gradient_norm(
                best["parameters"],
                best["gradient"],
            )

            say(
                f"Iteracija {used}/{MAX_ITERATIONS}"
                f" | proteklo {time.monotonic() - started:.1f} s"
                f" | cilj {best['value']:.9f}"
                f" | Max|proj.grad| {gradient_norm:.3e}"
            )

            save_checkpoint(
                path,
                best["parameters"],
                used,
                False,
                "Trening je u toku",
            )

    remaining = MAX_ITERATIONS - used

    if not done and remaining > 0:
        try:
            result = minimize(
                fun,
                best["parameters"],
                method="L-BFGS-B",
                jac=True,
                bounds=[
                    (
                        -PARAMETER_LIMIT,
                        PARAMETER_LIMIT,
                    )
                ] * PARAMETERS,
                callback=callback,
                options={
                    "maxiter": remaining,
                    "maxfun": MAX_FUNCTION_CALLS,
                    "maxls": 35,
                    "maxcor": 15,
                    "ftol": 1.0e-12,
                    "gtol": 1.0e-6,
                },
            )

            used = min(
                MAX_ITERATIONS,
                prior_iterations + int(result.nit),
            )

            message = str(result.message)

        except KeyboardInterrupt:
            save_checkpoint(
                path,
                best["parameters"],
                used,
                False,
                "Korisnički prekid",
            )

            raise

        except (
            RuntimeError,
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            message = (
                "Zadržan je najbolji validan kandidat. "
                f"Optimizacija je zaustavljena: {error}"
            )

    elif not done:
        message = "Iskorišćen je budžet iteracija."

    final_value, final_gradient = objective(
        best["parameters"],
        source,
        factors,
        targets,
        ridge,
    )

    gradient_norm = projected_gradient_norm(
        best["parameters"],
        final_gradient,
    )

    converged = (
        gradient_norm <= GRADIENT_LIMIT
    )

    save_checkpoint(
        path,
        best["parameters"],
        used,
        True,
        message,
    )

    say(
        f"Zadržani cilj: {final_value:.9f}",
        f"Max|proj.grad|: {gradient_norm:.6e}",
        f"Status optimizatora: {message}",
    )

    if not converged:
        say(
            "DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.",
            "Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.",
        )

    return Fitted(
        parameters=best["parameters"].copy(),
        loss=float(final_value),
        gradient_norm=gradient_norm,
        converged=converged,
        iterations=used,
        message=message,
    )


def evaluate(
    source,
    model,
    factors,
    targets,
    start,
    mode="memory",
    hits=False,
):
    require(
        len(factors) == len(targets)
        and 0 <= start < len(targets),
        "Neispravna evaluaciona granica.",
    )

    if mode == "memory":
        used_factors = factors
        reset = False

    elif mode == "reset":
        used_factors = factors
        reset = True

    elif mode == "no_input":
        used_factors = np.ones_like(factors)
        reset = False

    else:
        raise RuntimeError(
            "Nepoznata evaluaciona kontrola."
        )

    weights, _, _, _ = source.forward(
        model.parameters,
        used_factors,
        reset_memory=reset,
    )

    evaluated_weights = weights[start:]
    evaluated_targets = targets[start:]

    log_probability = subset_log_probability(
        evaluated_weights,
        evaluated_targets,
    )

    nll = -float(np.mean(log_probability))

    histogram = None

    if hits:
        selected = np.argsort(
            -evaluated_weights,
            axis=1,
            kind="stable",
        )[:, :PICKS]

        matches = np.sum(
            np.take_along_axis(
                evaluated_targets,
                selected,
                axis=1,
            ),
            axis=1,
        ).astype(np.int64)

        histogram = np.bincount(
            matches,
            minlength=PICKS + 1,
        )

        require(
            int(np.sum(histogram))
            == len(evaluated_targets),
            "Neispravna statistika pogodaka.",
        )

    return nll, histogram


def self_checks(source):
    say(
        "",
        "Pokrećem ugrađene provere pre treninga.",
    )

    weights = np.asarray(
        [[0.07, 0.13, 0.19, 0.23, 0.17, 0.21]],
        dtype=float,
    )

    picks = 3

    log_z, inclusion = subset_distribution(
        weights,
        picks=picks,
        marginals=True,
    )

    legal = list(
        combinations(range(6), picks)
    )

    raw = np.asarray(
        [
            np.prod(
                weights[0, list(subset)]
            )
            for subset in legal
        ]
    )

    enumerated_z = float(np.sum(raw))
    probabilities = raw / enumerated_z

    require(
        abs(
            float(log_z[0])
            - math.log(enumerated_z)
        ) < 1.0e-11,
        "Normalizacija odstupa od male enumeracije.",
    )

    expected_inclusion = np.zeros(6)

    for subset, probability in zip(
        legal,
        probabilities,
    ):
        expected_inclusion[
            list(subset)
        ] += probability

    require(
        np.max(
            np.abs(
                inclusion[0]
                - expected_inclusion
            )
        ) < 1.0e-11,
        "Marginale odstupaju od male enumeracije.",
    )

    selected, log_probability = exact_maximum(
        weights[0],
        picks=picks,
    )

    best_index = int(np.argmax(raw))

    require(
        selected == legal[best_index],
        "Maksimum odstupa od male enumeracije.",
    )

    require(
        abs(
            log_probability
            - math.log(
                float(
                    probabilities[best_index]
                )
            )
        ) < 1.0e-11,
        "Verovatnoća maksimuma je neispravna.",
    )

    for count in (0, 1, 6):
        _, marginal = subset_distribution(
            weights,
            picks=count,
            marginals=True,
        )

        require(
            abs(
                float(np.sum(marginal))
                - count
            ) < 1.0e-10,
            "Neispravan granični slučaj.",
        )

    tied, _ = exact_maximum(
        np.ones(6),
        picks=3,
    )

    require(
        tied == (0, 1, 2),
        "Izbor pri jednakim težinama nije deterministički.",
    )

    toy = np.zeros(
        (6, NUMBERS),
        dtype=float,
    )

    for row in range(len(toy)):
        numbers = (
            np.arange(PICKS) * 5
            + 3 * row
        ) % NUMBERS

        toy[row, numbers] = 1.0

    factors = source.input_factors(toy)

    parameters = (
        0.03
        * np.sin(
            np.arange(PARAMETERS) + 0.4
        )
    )

    train_factors = factors[:-1]
    targets = toy[1:]

    _, analytic = objective(
        parameters,
        source,
        train_factors,
        targets,
        0.1,
    )

    epsilon = 1.0e-6
    largest_error = 0.0

    # Proverava se svih 30 parametara.
    for index in range(PARAMETERS):
        direction = np.zeros(PARAMETERS)
        direction[index] = epsilon

        plus, _ = objective(
            parameters + direction,
            source,
            train_factors,
            targets,
            0.1,
        )

        minus, _ = objective(
            parameters - direction,
            source,
            train_factors,
            targets,
            0.1,
        )

        numerical = (
            plus - minus
        ) / (2.0 * epsilon)

        error = abs(
            numerical - analytic[index]
        )

        largest_error = max(
            largest_error,
            error,
        )

        require(
            error
            < 5.0e-5
            * (1.0 + abs(analytic[index])),
            "Gradijent kroz kvantnu memoriju ne prolazi proveru.",
        )

    _, states, hamiltonian, unitary = source.forward(
        parameters,
        factors,
    )

    qiskit_error = source.verify_qiskit_history(
        parameters,
        factors,
        states[-1],
    )

    _, prefix_states, _, _ = source.forward(
        parameters,
        factors[:3],
    )

    require(
        np.max(
            np.abs(
                prefix_states - states[:4]
            )
        ) < 1.0e-12,
        "Budući ulazi menjaju ranija stanja.",
    )

    before = (
        states[-2] * factors[-1]
    )

    transport = verify_current_transport(
        hamiltonian,
        before,
        states[-1],
    )

    finite_difference_step = 1.0e-6

    plus = np.asarray(
        Statevector(before).evolve(
            Operator(
                HamiltonianGate(
                    hamiltonian,
                    time=finite_difference_step,
                )
            )
        ).data
    )

    minus = np.asarray(
        Statevector(before).evolve(
            Operator(
                HamiltonianGate(
                    hamiltonian,
                    time=-finite_difference_step,
                )
            )
        ).data
    )

    numerical_density_derivative = (
        np.abs(plus) ** 2
        - np.abs(minus) ** 2
    ) / (2.0 * finite_difference_step)

    current, _, _ = current_and_guidance(
        hamiltonian,
        before,
    )

    require(
        np.max(
            np.abs(
                numerical_density_derivative
                - np.sum(current, axis=1)
            )
        ) < 2.0e-6,
        "Struje odstupaju od izvoda Qiskit verovatnoća.",
    )

    seven = source.seven_register_step(
        parameters,
        factors[-1],
    )

    require(
        seven.num_qubits == 42,
        "Neispravna konstrukcija sedam registara.",
    )

    require(
        np.max(
            np.abs(
                unitary @ before - states[-1]
            )
        ) < 1.0e-12,
        "Poslednji korak memorije nije usaglašen.",
    )

    say(
        "Normalizacija, margine i maksimum: provereni na maloj enumeraciji.",
        f"Najveća greška gradijenta: {largest_error:.3e}",
        f"Greška pune Qiskit istorije: {qiskit_error:.3e}",
        f"Greška zakona vođenja: {transport['guidance_error']:.3e}",
        f"Greška transporta gustine: {transport['transport_error']:.3e}",
        "Hronološka uzročnost: potvrđena na provernom primeru.",
        "Sedam registara: konstrukcija od 42 kubita.",
    )


def longest_run(numbers):
    longest = 0
    current = 0
    previous = None

    for number in sorted(numbers):
        if (
            previous is not None
            and number == previous + 1
        ):
            current += 1
        else:
            current = 1

        longest = max(longest, current)
        previous = number

    return longest


def print_result(result):
    say(
        "",
        result["name"],
        "NEXT: " + " ".join(
            str(number + 1)
            for number in result["selected"]
        ),
        f"Validacioni NLL: {result['validation_nll']:.9f}",
        f"Odvojeni test NLL: {result['test_nll']:.9f}",
        f"Test NLL sa resetovanom memorijom: {result['reset_nll']:.9f}",
        f"Test NLL bez istorijskih ulaznih faza: {result['no_input_nll']:.9f}",
        f"Modelna verovatnoća NEXT: {result['next_probability']:.12g}",
        f"Normalizacioni odnos: {result['normalization_ratio']:.12f}",
        f"Greška pune Qiskit istorije: {result['qiskit_error']:.3e}",
        f"Greška diskretnog zakona vođenja: {result['guidance_error']:.3e}",
        f"Greška integrisanog transporta: {result['transport_error']:.3e}",
        f"Greška očuvanja mase transporta: {result['transport_mass_error']:.3e}",
        f"Red kvadrature transporta: {result['quadrature_order']}",
        f"Efekat memorije na NEXT margine: {result['memory_effect']:.6e}",
        f"Osetljivost poslednjih raspodela: {result['sensitivity']:.6e}",
        f"Zbir marginalnih verovatnoća: {result['marginal_sum']:.12f}",
        f"Završni Max|proj.grad|: {result['gradient_norm']:.6e}",
        "Test: broj kola prema broju pogodaka",
        " | ".join(
            f"{hits}: {int(count)}"
            for hits, count in enumerate(
                result["histogram"]
            )
        ),
        f"Prosečan broj pogodaka na testu: {result['mean_hits']:.6f}",
        "Marginalne verovatnoće uključivanja brojeva:",
    )

    inclusion = result["inclusion"]

    for start in range(0, NUMBERS, 6):
        say(
            " | ".join(
                f"{index + 1:02d}: {inclusion[index]:.8f}"
                for index in range(
                    start,
                    min(start + 6, NUMBERS),
                )
            )
        )

    for warning in result["warnings"]:
        say(f"DIJAGNOSTIKA: {warning}")


def process_file(source, path, draws, occupancy):
    all_factors = source.input_factors(
        occupancy
    )

    factors = all_factors[:-1]
    targets = occupancy[1:]
    count = len(targets)

    split_60 = int(0.60 * count)
    split_70 = int(0.70 * count)
    split_80 = int(0.80 * count)

    require(
        20 < split_60
        < split_70
        < split_80
        < count,
        "Neispravne hronološke podele.",
    )

    folds = (
        (split_60, split_70),
        (split_70, split_80),
    )

    validation_records = []
    unconfirmed_fits = 0

    for ridge in RIDGES:
        weighted_nll = 0.0
        validation_count = 0

        warm = source.initial_parameters()

        for fold_index, (train_end, validation_end) in enumerate(
            folds,
            start=1,
        ):
            model = fit_model(
                source,
                factors[:train_end],
                targets[:train_end],
                ridge,
                (
                    f"{path.name} / validacija {fold_index}"
                    f" / ridge {ridge}"
                ),
                initial=warm,
            )

            if not model.converged:
                unconfirmed_fits += 1

            fold_nll, _ = evaluate(
                source,
                model,
                factors[:validation_end],
                targets[:validation_end],
                start=train_end,
            )

            fold_count = (
                validation_end - train_end
            )

            weighted_nll += (
                fold_nll * fold_count
            )

            validation_count += fold_count
            warm = model.parameters.copy()

            say(
                f"Validacija {fold_index} NLL: {fold_nll:.9f}"
            )

        mean_nll = (
            weighted_nll / validation_count
        )

        validation_records.append(
            (
                float(mean_nll),
                float(ridge),
                warm.copy(),
            )
        )

        say(
            f"Objedinjeni validacioni NLL za ridge {ridge}: "
            f"{mean_nll:.9f}"
        )

    validation_records.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    validation_nll, chosen_ridge, warm = (
        validation_records[0]
    )

    say(
        "",
        f"{path.name}: izabrani ridge {chosen_ridge}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit_model(
        source,
        factors[:split_80],
        targets[:split_80],
        chosen_ridge,
        f"{path.name} / model za odvojeni test",
        initial=warm,
    )

    if not test_model.converged:
        unconfirmed_fits += 1

    test_nll, histogram = evaluate(
        source,
        test_model,
        factors,
        targets,
        start=split_80,
        mode="memory",
        hits=True,
    )

    reset_nll, _ = evaluate(
        source,
        test_model,
        factors,
        targets,
        start=split_80,
        mode="reset",
    )

    no_input_nll, _ = evaluate(
        source,
        test_model,
        factors,
        targets,
        start=split_80,
        mode="no_input",
    )

    say(
        "",
        f"{path.name}: odvojeni test je završen.",
        f"Test NLL: {test_nll:.9f}",
        "Parametri tokom testa ostaju zamrznuti.",
        "Svaki naredni test koristi samo tada poznata prethodna kola.",
        "Kontrole nisu posebno trenirani konkurenti.",
    )

    final_model = fit_model(
        source,
        factors,
        targets,
        chosen_ridge,
        f"{path.name} / završno učenje na svim prelazima",
        initial=test_model.parameters,
    )

    if not final_model.converged:
        unconfirmed_fits += 1

    all_weights, states, hamiltonian, unitary = source.forward(
        final_model.parameters,
        all_factors,
    )

    next_weights = all_weights[-1]

    selected, next_log_probability = exact_maximum(
        next_weights
    )

    require(
        len(selected) == PICKS
        and len(set(selected)) == PICKS
        and all(
            0 <= number < NUMBERS
            for number in selected
        ),
        "NEXT nije legalna sedmorka.",
    )

    log_z, inclusion = subset_distribution(
        next_weights[None, :],
        marginals=True,
    )

    alternative_log_z = alternative_log_normalizer(
        next_weights
    )

    require(
        abs(
            alternative_log_z
            - float(log_z[0])
        ) < 2.0e-10,
        "Dve normalizacije NEXT raspodele se ne slažu.",
    )

    normalization_ratio = math.exp(
        alternative_log_z
        - float(log_z[0])
    )

    say(
        "",
        f"{path.name}: proveravam punu Qiskit istoriju i struje.",
    )

    qiskit_error = source.verify_qiskit_history(
        final_model.parameters,
        all_factors,
        states[-1],
    )

    before_evolution = (
        states[-2] * all_factors[-1]
    )

    require(
        np.max(
            np.abs(
                np.abs(before_evolution) ** 2
                - np.abs(states[-2]) ** 2
            )
        ) < 2.0e-10,
        "Ulazna fazna kapija menja gustinu umesto faze.",
    )

    transport = verify_current_transport(
        hamiltonian,
        before_evolution,
        states[-1],
    )

    source.seven_register_step(
        final_model.parameters,
        all_factors[-1],
    )

    # Provera poslednjeg koraka za sedam jednakih registara.
    for _register in range(PICKS):
        reference = Statevector(
            states[-2].copy()
        )

        reference = reference.evolve(
            Operator(
                np.diag(all_factors[-1])
            )
        )

        reference = reference.evolve(
            Operator(unitary)
        )

        require(
            np.max(
                np.abs(
                    reference.data - states[-1]
                )
            ) < 2.0e-10,
            "Poslednji korak registra odstupa od Qiskit simulacije.",
        )

    reset_state = (
        unitary
        @ (
            source.initial_state
            * all_factors[-1]
        )
    )

    reset_weights = (
        np.abs(
            reset_state[:NUMBERS]
        ) ** 2
    )

    _, reset_inclusion = subset_distribution(
        reset_weights[None, :],
        marginals=True,
    )

    memory_effect = float(
        np.max(
            np.abs(
                inclusion[0]
                - reset_inclusion[0]
            )
        )
    )

    recent = all_weights[
        -min(8, len(all_weights)):
    ]

    _, recent_inclusion = subset_distribution(
        recent,
        marginals=True,
    )

    sensitivity = float(
        np.max(
            np.ptp(
                recent_inclusion,
                axis=0,
            )
        )
    )

    warnings = []

    if unconfirmed_fits:
        warnings.append(
            f"{unconfirmed_fits} treninga nema potvrđen gradijentni uslov; "
            "njihovi minimumi nisu potvrđeni."
        )

    if test_nll >= reset_nll:
        warnings.append(
            "Memorija nije dala niži test NLL od dijagnostičkog resetovanja."
        )

    if test_nll >= no_input_nll:
        warnings.append(
            "Istorijske ulazne faze nisu dale niži test NLL "
            "od dijagnostičke kontrole bez tih faza."
        )

    if memory_effect < 1.0e-8:
        warnings.append(
            "Uklanjanje memorije malo menja NEXT raspodelu."
        )

    if sensitivity < 1.0e-8:
        warnings.append(
            "Poslednje raspodele slabo se razlikuju."
        )

    run = longest_run(selected)

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if selected[-1] - selected[0] <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    selected_array = np.asarray(
        selected,
        dtype=np.int64,
    )

    if np.any(
        np.all(
            draws == selected_array[None, :],
            axis=1,
        )
    ):
        warnings.append(
            "NEXT se već pojavio u istoriji ovog CSV-a."
        )

    if np.array_equal(
        selected_array,
        draws[-1],
    ):
        warnings.append(
            "NEXT je jednak poslednjem kolu."
        )

    mean_hits = float(
        np.dot(
            np.arange(PICKS + 1),
            histogram,
        ) / np.sum(histogram)
    )

    result = {
        "name": path.name,
        "selected": selected,
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "reset_nll": reset_nll,
        "no_input_nll": no_input_nll,
        "next_probability": math.exp(
            next_log_probability
        ),
        "normalization_ratio": normalization_ratio,
        "qiskit_error": qiskit_error,
        "inclusion": inclusion[0].copy(),
        "marginal_sum": float(
            np.sum(inclusion[0])
        ),
        "memory_effect": memory_effect,
        "sensitivity": sensitivity,
        "gradient_norm": final_model.gradient_norm,
        "histogram": histogram,
        "mean_hits": mean_hits,
        "warnings": warnings,
        **transport,
    }

    print_result(result)

    return result


def main():
    say(
        "QUANTUM BOHM V1 — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "",
        "Diskretna adaptacija inspirisana Bohmom.",
        "Kvantna memorija prenosi kompleksno stanje kroz istoriju.",
        "Prethodno kolo menja faze kvantnog stanja.",
        "Učeni Hamiltonijan određuje sledeću kvantnu evoluciju.",
        "Evolucija koristi Qiskit HamiltonianGate.",
        "Završna istorija proverava se pomoću Qiskit Statevector.",
        "",
        "Struje verovatnoće zadovoljavaju jednačinu kontinuiteta.",
        "Diskretni zakon vođenja proverava se preko stopa transporta.",
        "Računa se transport raspodele, bez uzorkovanja putanja.",
        "Ovo nije doslovna Bohmova mehanika čestica u fizičkom prostoru.",
        "Brojevi su oznake stanja, ne fizičke koordinate.",
        "",
        "Sedam registara koristi po šest kubita i isto memorijsko stanje.",
        "Proračun koristi kompaktnu proizvodnu reprezentaciju.",
        "Nema međuregistarskog kvantnog sprezanja.",
        "Raspodela sedmorki je proizvod kvantnih težina",
        "uslovljen izborom sedam različitih brojeva iz 1–39.",
        "Normalizacija i maksimum te raspodele računaju se tačno.",
        "",
        "Hamiltonijan se uči na svim prelazima završnog treninga.",
        "Gradijent prolazi kroz celu istoriju bez vremenskog odsecanja.",
        "Parametri se biraju hronološkom validacijom.",
        "Odvojeni test ne bira parametre.",
        "Završni model koristi sve dostupne istorijske prelaze.",
        "",
        "Optimizator, gradijent i normalizacija su klasični.",
        "Nema klasičnog regresora koji zamenjuje kvantne amplitude.",
        "Nema slučajnog generisanja, šuma ili uzorkovanja merenja.",
        "Nema kozmetičkih filtera kombinacija.",
        "Optimizacija je lokalna; globalni minimum nije potvrđen.",
        "",
        "Kontrole resetuju memoriju ili uklanjaju ulazne faze.",
        "Kontrole nisu posebno trenirani konkurenti.",
        "Niži NLL znači bolju dodeljenu verovatnoću stvarnim sedmorkama.",
        "",
        f"Kontrolne tačke: {CHECKPOINT_DIRECTORY}",
    )

    source = BohmMemory()

    loaded = []

    for path in FILES:
        draws, occupancy = load_draws(path)

        loaded.append(
            (path, draws, occupancy)
        )

    self_checks(source)

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for path, draws, occupancy in loaded:
        results.append(
            process_file(
                source,
                path,
                draws,
                occupancy,
            )
        )

    require(
        len(results) == len(FILES),
        "Nisu završena sva tri ulaza.",
    )

    say(
        "",
        "ZAVRŠNI NEXT REZULTATI",
    )

    for result in results:
        say(
            result["name"]
            + " | NEXT: "
            + " ".join(
                str(number + 1)
                for number in result["selected"]
            )
        )

    for first in range(len(results)):
        for second in range(
            first + 1,
            len(results),
        ):
            if (
                results[first]["selected"]
                == results[second]["selected"]
            ):
                say(
                    "DIJAGNOSTIKA: isti NEXT za "
                    f"{results[first]['name']} i "
                    f"{results[second]['name']}.",
                    (
                        "Efekti memorije: "
                        f"{results[first]['memory_effect']:.6e} i "
                        f"{results[second]['memory_effect']:.6e}."
                    ),
                    "Rezultat nije menjan radi veštačkog razlikovanja.",
                )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say(
            "",
            "PREKINUTO na zahtev korisnika.",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM BOHM V1 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4

Diskretna adaptacija inspirisana Bohmom.
Kvantna memorija prenosi kompleksno stanje kroz istoriju.
Prethodno kolo menja faze kvantnog stanja.
Učeni Hamiltonijan određuje sledeću kvantnu evoluciju.
Evolucija koristi Qiskit HamiltonianGate.
Završna istorija proverava se pomoću Qiskit Statevector.

Struje verovatnoće zadovoljavaju jednačinu kontinuiteta.
Diskretni zakon vođenja proverava se preko stopa transporta.
Računa se transport raspodele, bez uzorkovanja putanja.
Ovo nije doslovna Bohmova mehanika čestica u fizičkom prostoru.
Brojevi su oznake stanja, ne fizičke koordinate.

Sedam registara koristi po šest kubita i isto memorijsko stanje.
Proračun koristi kompaktnu proizvodnu reprezentaciju.
Nema međuregistarskog kvantnog sprezanja.
Raspodela sedmorki je proizvod kvantnih težina
uslovljen izborom sedam različitih brojeva iz 1–39.
Normalizacija i maksimum te raspodele računaju se tačno.

Hamiltonijan se uči na svim prelazima završnog treninga.
Gradijent prolazi kroz celu istoriju bez vremenskog odsecanja.
Parametri se biraju hronološkom validacijom.
Odvojeni test ne bira parametre.
Završni model koristi sve dostupne istorijske prelaze.

Optimizator, gradijent i normalizacija su klasični.
Nema klasičnog regresora koji zamenjuje kvantne amplitude.
Nema slučajnog generisanja, šuma ili uzorkovanja merenja.
Nema kozmetičkih filtera kombinacija.
Optimizacija je lokalna; globalni minimum nije potvrđen.

Kontrole resetuju memoriju ili uklanjaju ulazne faze.
Kontrole nisu posebno trenirani konkurenti.
Niži NLL znači bolju dodeljenu verovatnoću stvarnim sedmorkama.

Kontrolne tačke: /Quantum_Bohm/Quantum_Bohm_V1_checkpoints

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
CSV redosled ostaje od najstarijeg do najnovijeg kola.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
CSV redosled ostaje od najstarijeg do najnovijeg kola.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
CSV redosled ostaje od najstarijeg do najnovijeg kola.

Pokrećem ugrađene provere pre treninga.
Normalizacija, margine i maksimum: provereni na maloj enumeraciji.
Najveća greška gradijenta: 5.732e-09
Greška pune Qiskit istorije: 5.551e-17
Greška zakona vođenja: 2.082e-17
Greška transporta gustine: 1.128e-16
Hronološka uzročnost: potvrđena na provernom primeru.
Sedam registara: konstrukcija od 42 kubita.

loto7_4710_k86.csv / validacija 1 / ridge 0.03
Trening prelaza: 2825 | ridge: 0.03
Iteracija 5/120 | proteklo 1.5 s | cilj 19.772781287 | Max|proj.grad| 1.890e+01
Iteracija 10/120 | proteklo 2.1 s | cilj 19.699549362 | Max|proj.grad| 8.489e+00
Iteracija 15/120 | proteklo 2.7 s | cilj 19.647601773 | Max|proj.grad| 5.584e+00
Iteracija 20/120 | proteklo 3.7 s | cilj 19.626239193 | Max|proj.grad| 5.097e+00
Iteracija 25/120 | proteklo 4.1 s | cilj 19.620504873 | Max|proj.grad| 7.472e-01
Iteracija 30/120 | proteklo 4.6 s | cilj 19.614725131 | Max|proj.grad| 1.691e+00
Iteracija 35/120 | proteklo 5.1 s | cilj 19.610213654 | Max|proj.grad| 1.297e+00
Iteracija 40/120 | proteklo 5.6 s | cilj 19.605222638 | Max|proj.grad| 1.120e+00
Iteracija 45/120 | proteklo 6.1 s | cilj 19.602909454 | Max|proj.grad| 1.249e+00
Iteracija 50/120 | proteklo 6.8 s | cilj 19.598714459 | Max|proj.grad| 9.429e-01
Iteracija 55/120 | proteklo 7.4 s | cilj 19.595340858 | Max|proj.grad| 7.301e-01
Iteracija 60/120 | proteklo 8.0 s | cilj 19.591502124 | Max|proj.grad| 1.368e+00
Iteracija 65/120 | proteklo 8.5 s | cilj 19.586075484 | Max|proj.grad| 8.250e-01
Iteracija 70/120 | proteklo 9.0 s | cilj 19.583128383 | Max|proj.grad| 1.894e+00
Iteracija 75/120 | proteklo 9.6 s | cilj 19.579111576 | Max|proj.grad| 4.678e-01
Iteracija 80/120 | proteklo 10.1 s | cilj 19.578243668 | Max|proj.grad| 5.705e-01
Iteracija 85/120 | proteklo 10.7 s | cilj 19.575836964 | Max|proj.grad| 5.060e-01
Iteracija 90/120 | proteklo 11.2 s | cilj 19.571918616 | Max|proj.grad| 2.298e+00
Iteracija 95/120 | proteklo 11.7 s | cilj 19.567187955 | Max|proj.grad| 1.344e+00
Iteracija 100/120 | proteklo 12.4 s | cilj 19.562280372 | Max|proj.grad| 1.190e+00
Iteracija 105/120 | proteklo 12.9 s | cilj 19.561727083 | Max|proj.grad| 3.087e-01
Iteracija 110/120 | proteklo 13.9 s | cilj 19.553407458 | Max|proj.grad| 5.640e+00
Iteracija 115/120 | proteklo 14.5 s | cilj 19.542827827 | Max|proj.grad| 2.362e+00
Iteracija 120/120 | proteklo 15.1 s | cilj 19.537935390 | Max|proj.grad| 1.003e+00
Zadržani cilj: 19.537935390
Max|proj.grad|: 1.002909e+00
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 19.919741478

loto7_4710_k86.csv / validacija 2 / ridge 0.03
Trening prelaza: 3296 | ridge: 0.03
Iteracija 5/120 | proteklo 1.5 s | cilj 19.584749276 | Max|proj.grad| 1.496e+00
Iteracija 10/120 | proteklo 2.3 s | cilj 19.575523526 | Max|proj.grad| 2.948e+00
Iteracija 15/120 | proteklo 2.9 s | cilj 19.570878191 | Max|proj.grad| 7.276e-01
Iteracija 20/120 | proteklo 3.6 s | cilj 19.567763971 | Max|proj.grad| 2.842e+00
Iteracija 25/120 | proteklo 4.1 s | cilj 19.561679503 | Max|proj.grad| 1.234e+00
Iteracija 30/120 | proteklo 4.7 s | cilj 19.559872929 | Max|proj.grad| 1.515e+00
Iteracija 35/120 | proteklo 5.3 s | cilj 19.557825150 | Max|proj.grad| 4.297e-01
Iteracija 40/120 | proteklo 5.8 s | cilj 19.555956194 | Max|proj.grad| 8.331e-01
Iteracija 45/120 | proteklo 6.4 s | cilj 19.555458668 | Max|proj.grad| 3.463e-01
Iteracija 50/120 | proteklo 7.3 s | cilj 19.554190289 | Max|proj.grad| 1.160e+00
Iteracija 55/120 | proteklo 7.9 s | cilj 19.553327369 | Max|proj.grad| 2.003e-01
Iteracija 60/120 | proteklo 8.5 s | cilj 19.551945825 | Max|proj.grad| 8.418e-01
Iteracija 65/120 | proteklo 9.2 s | cilj 19.550472867 | Max|proj.grad| 2.829e-01
Iteracija 70/120 | proteklo 9.7 s | cilj 19.549301638 | Max|proj.grad| 9.760e-01
Iteracija 75/120 | proteklo 10.3 s | cilj 19.545623217 | Max|proj.grad| 2.173e+00
Iteracija 80/120 | proteklo 11.0 s | cilj 19.542459478 | Max|proj.grad| 5.106e-01
Iteracija 85/120 | proteklo 11.6 s | cilj 19.540932855 | Max|proj.grad| 4.761e-01
Iteracija 90/120 | proteklo 12.5 s | cilj 19.539040974 | Max|proj.grad| 2.709e+00
Iteracija 95/120 | proteklo 13.1 s | cilj 19.535858006 | Max|proj.grad| 2.711e+00
Iteracija 100/120 | proteklo 13.9 s | cilj 19.532009560 | Max|proj.grad| 1.961e+00
Iteracija 105/120 | proteklo 14.6 s | cilj 19.529357321 | Max|proj.grad| 6.509e-01
Iteracija 110/120 | proteklo 15.1 s | cilj 19.528541410 | Max|proj.grad| 5.086e-01
Iteracija 115/120 | proteklo 15.7 s | cilj 19.527713334 | Max|proj.grad| 3.355e-01
Iteracija 120/120 | proteklo 16.3 s | cilj 19.526685844 | Max|proj.grad| 1.017e+00
Zadržani cilj: 19.526685844
Max|proj.grad|: 1.017280e+00
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 20.115702709
Objedinjeni validacioni NLL za ridge 0.03: 20.017722094

loto7_4710_k86.csv / validacija 1 / ridge 0.3
Trening prelaza: 2825 | ridge: 0.3
Iteracija 5/120 | proteklo 1.4 s | cilj 19.750270629 | Max|proj.grad| 8.277e+00
Iteracija 10/120 | proteklo 2.1 s | cilj 19.710876176 | Max|proj.grad| 5.873e+00
Iteracija 15/120 | proteklo 2.9 s | cilj 19.688130152 | Max|proj.grad| 6.105e+00
Iteracija 20/120 | proteklo 3.6 s | cilj 19.675869417 | Max|proj.grad| 6.654e+00
Iteracija 25/120 | proteklo 4.6 s | cilj 19.658251880 | Max|proj.grad| 2.281e+00
Iteracija 30/120 | proteklo 5.1 s | cilj 19.650883484 | Max|proj.grad| 1.681e+00
Iteracija 35/120 | proteklo 5.6 s | cilj 19.647218516 | Max|proj.grad| 1.841e+00
Iteracija 40/120 | proteklo 6.3 s | cilj 19.638657611 | Max|proj.grad| 5.790e+00
Iteracija 45/120 | proteklo 7.0 s | cilj 19.627160578 | Max|proj.grad| 3.906e+00
Iteracija 50/120 | proteklo 7.7 s | cilj 19.606886908 | Max|proj.grad| 3.462e+00
Iteracija 55/120 | proteklo 8.1 s | cilj 19.600084222 | Max|proj.grad| 2.818e+00
Iteracija 60/120 | proteklo 8.7 s | cilj 19.595852177 | Max|proj.grad| 4.593e-01
Iteracija 65/120 | proteklo 9.3 s | cilj 19.593809386 | Max|proj.grad| 1.201e+00
Iteracija 70/120 | proteklo 9.9 s | cilj 19.591004659 | Max|proj.grad| 1.508e+00
Iteracija 75/120 | proteklo 10.4 s | cilj 19.590031971 | Max|proj.grad| 5.916e-01
Iteracija 80/120 | proteklo 10.8 s | cilj 19.589336417 | Max|proj.grad| 1.077e+00
Iteracija 85/120 | proteklo 11.5 s | cilj 19.587632163 | Max|proj.grad| 5.377e-01
Iteracija 90/120 | proteklo 12.0 s | cilj 19.586113377 | Max|proj.grad| 7.228e-01
Iteracija 95/120 | proteklo 12.7 s | cilj 19.579854662 | Max|proj.grad| 1.793e+00
Iteracija 100/120 | proteklo 13.2 s | cilj 19.578132096 | Max|proj.grad| 2.669e-01
Iteracija 105/120 | proteklo 13.8 s | cilj 19.576765154 | Max|proj.grad| 7.862e-01
Iteracija 110/120 | proteklo 14.4 s | cilj 19.574512090 | Max|proj.grad| 1.916e+00
Iteracija 115/120 | proteklo 14.9 s | cilj 19.572951693 | Max|proj.grad| 5.255e-01
Iteracija 120/120 | proteklo 15.4 s | cilj 19.571116293 | Max|proj.grad| 1.147e+00
Zadržani cilj: 19.571116293
Max|proj.grad|: 1.146741e+00
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 19.878959231

loto7_4710_k86.csv / validacija 2 / ridge 0.3
Trening prelaza: 3296 | ridge: 0.3
Iteracija 5/120 | proteklo 1.6 s | cilj 19.607856194 | Max|proj.grad| 2.538e+00
Iteracija 10/120 | proteklo 2.1 s | cilj 19.603223691 | Max|proj.grad| 2.776e+00
Iteracija 15/120 | proteklo 2.7 s | cilj 19.600155518 | Max|proj.grad| 8.122e-01
Iteracija 20/120 | proteklo 3.2 s | cilj 19.598168374 | Max|proj.grad| 1.018e+00
Iteracija 25/120 | proteklo 3.9 s | cilj 19.596027781 | Max|proj.grad| 1.060e+00
Iteracija 30/120 | proteklo 4.7 s | cilj 19.593512765 | Max|proj.grad| 1.508e+00
Iteracija 35/120 | proteklo 5.3 s | cilj 19.591868657 | Max|proj.grad| 3.654e-01
Iteracija 40/120 | proteklo 5.9 s | cilj 19.590645872 | Max|proj.grad| 6.882e-01
Iteracija 45/120 | proteklo 6.4 s | cilj 19.589555115 | Max|proj.grad| 7.298e-01
Iteracija 50/120 | proteklo 7.0 s | cilj 19.588764709 | Max|proj.grad| 4.053e-01
Iteracija 55/120 | proteklo 7.5 s | cilj 19.587715710 | Max|proj.grad| 7.176e-01
Iteracija 60/120 | proteklo 8.1 s | cilj 19.586891940 | Max|proj.grad| 7.444e-01
Iteracija 65/120 | proteklo 9.0 s | cilj 19.585879184 | Max|proj.grad| 1.357e+00
Iteracija 70/120 | proteklo 9.5 s | cilj 19.583524525 | Max|proj.grad| 6.661e-01
Iteracija 75/120 | proteklo 10.1 s | cilj 19.582330905 | Max|proj.grad| 1.232e+00
Iteracija 80/120 | proteklo 10.9 s | cilj 19.579848055 | Max|proj.grad| 3.234e+00
Iteracija 85/120 | proteklo 11.6 s | cilj 19.571539206 | Max|proj.grad| 1.702e+00
Iteracija 90/120 | proteklo 12.3 s | cilj 19.567684967 | Max|proj.grad| 2.129e+00
Iteracija 95/120 | proteklo 12.9 s | cilj 19.565199435 | Max|proj.grad| 3.247e-01
Iteracija 100/120 | proteklo 13.7 s | cilj 19.563646262 | Max|proj.grad| 1.751e+00
Iteracija 105/120 | proteklo 14.3 s | cilj 19.559373090 | Max|proj.grad| 1.580e+00
Iteracija 110/120 | proteklo 14.9 s | cilj 19.557956389 | Max|proj.grad| 3.436e-01
Iteracija 115/120 | proteklo 15.5 s | cilj 19.557570147 | Max|proj.grad| 1.818e-01
Iteracija 120/120 | proteklo 16.1 s | cilj 19.557382620 | Max|proj.grad| 1.758e-01
Zadržani cilj: 19.557382620
Max|proj.grad|: 1.758055e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 20.173925651
Objedinjeni validacioni NLL za ridge 0.3: 20.026442441

loto7_4710_k86.csv: izabrani ridge 0.03
Validacioni NLL: 20.017722094

loto7_4710_k86.csv / model za odvojeni test
Trening prelaza: 3767 | ridge: 0.03
Iteracija 5/120 | proteklo 1.7 s | cilj 19.591619236 | Max|proj.grad| 1.792e+00
Iteracija 10/120 | proteklo 2.3 s | cilj 19.590444812 | Max|proj.grad| 5.352e-01
Iteracija 15/120 | proteklo 3.0 s | cilj 19.588648136 | Max|proj.grad| 6.229e-01
Iteracija 20/120 | proteklo 3.6 s | cilj 19.588127402 | Max|proj.grad| 6.183e-01
Iteracija 25/120 | proteklo 4.3 s | cilj 19.587204310 | Max|proj.grad| 3.915e-01
Iteracija 30/120 | proteklo 4.9 s | cilj 19.586751193 | Max|proj.grad| 2.837e-01
Iteracija 35/120 | proteklo 5.6 s | cilj 19.586043315 | Max|proj.grad| 5.560e-01
Iteracija 40/120 | proteklo 6.2 s | cilj 19.585675165 | Max|proj.grad| 2.487e-01
Iteracija 45/120 | proteklo 7.0 s | cilj 19.584841037 | Max|proj.grad| 5.798e-01
Iteracija 50/120 | proteklo 7.6 s | cilj 19.584236452 | Max|proj.grad| 5.921e-01
Iteracija 55/120 | proteklo 8.7 s | cilj 19.581838560 | Max|proj.grad| 3.374e+00
Iteracija 60/120 | proteklo 9.6 s | cilj 19.574954842 | Max|proj.grad| 2.005e+00
Iteracija 65/120 | proteklo 10.2 s | cilj 19.572743080 | Max|proj.grad| 1.106e+00
Iteracija 70/120 | proteklo 10.9 s | cilj 19.570868957 | Max|proj.grad| 1.427e+00
Iteracija 75/120 | proteklo 11.8 s | cilj 19.569295061 | Max|proj.grad| 1.015e+00
Iteracija 80/120 | proteklo 12.4 s | cilj 19.568233387 | Max|proj.grad| 2.258e-01
Iteracija 85/120 | proteklo 13.1 s | cilj 19.567964363 | Max|proj.grad| 3.405e-01
Iteracija 90/120 | proteklo 13.7 s | cilj 19.567177131 | Max|proj.grad| 2.174e-01
Iteracija 95/120 | proteklo 14.4 s | cilj 19.566596618 | Max|proj.grad| 3.737e-01
Iteracija 100/120 | proteklo 15.4 s | cilj 19.565989960 | Max|proj.grad| 6.511e-01
Iteracija 105/120 | proteklo 16.0 s | cilj 19.565138856 | Max|proj.grad| 4.661e-01
Iteracija 110/120 | proteklo 16.6 s | cilj 19.564735245 | Max|proj.grad| 2.997e-01
Iteracija 115/120 | proteklo 17.3 s | cilj 19.564465539 | Max|proj.grad| 1.932e-01
Iteracija 120/120 | proteklo 17.9 s | cilj 19.564341935 | Max|proj.grad| 2.005e-01
Zadržani cilj: 19.564341935
Max|proj.grad|: 2.005206e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.

loto7_4710_k86.csv: odvojeni test je završen.
Test NLL: 19.818428365
Parametri tokom testa ostaju zamrznuti.
Svaki naredni test koristi samo tada poznata prethodna kola.
Kontrole nisu posebno trenirani konkurenti.

loto7_4710_k86.csv / završno učenje na svim prelazima
Trening prelaza: 4709 | ridge: 0.03
Iteracija 5/120 | proteklo 2.0 s | cilj 19.605751184 | Max|proj.grad| 8.262e-01
Iteracija 10/120 | proteklo 2.8 s | cilj 19.604820605 | Max|proj.grad| 9.615e-01
Iteracija 15/120 | proteklo 3.6 s | cilj 19.602160550 | Max|proj.grad| 1.267e+00
Iteracija 20/120 | proteklo 4.7 s | cilj 19.599180588 | Max|proj.grad| 1.982e+00
Iteracija 25/120 | proteklo 5.5 s | cilj 19.597202785 | Max|proj.grad| 9.952e-01
Iteracija 30/120 | proteklo 6.5 s | cilj 19.595737537 | Max|proj.grad| 5.168e-01
Iteracija 35/120 | proteklo 7.2 s | cilj 19.595283072 | Max|proj.grad| 7.917e-01
Iteracija 40/120 | proteklo 8.0 s | cilj 19.594401296 | Max|proj.grad| 2.219e-01
Iteracija 45/120 | proteklo 8.8 s | cilj 19.594305621 | Max|proj.grad| 2.870e-01
Iteracija 50/120 | proteklo 9.6 s | cilj 19.593884581 | Max|proj.grad| 2.956e-01
Iteracija 55/120 | proteklo 10.4 s | cilj 19.593782128 | Max|proj.grad| 1.691e-01
Iteracija 60/120 | proteklo 11.3 s | cilj 19.593375608 | Max|proj.grad| 4.100e-01
Iteracija 65/120 | proteklo 12.0 s | cilj 19.592875753 | Max|proj.grad| 1.297e-01
Iteracija 70/120 | proteklo 12.8 s | cilj 19.592806828 | Max|proj.grad| 1.067e-01
Iteracija 75/120 | proteklo 13.6 s | cilj 19.592410000 | Max|proj.grad| 3.011e-01
Iteracija 80/120 | proteklo 14.4 s | cilj 19.592176263 | Max|proj.grad| 5.545e-01
Iteracija 85/120 | proteklo 15.2 s | cilj 19.591873460 | Max|proj.grad| 4.032e-01
Iteracija 90/120 | proteklo 16.1 s | cilj 19.589996924 | Max|proj.grad| 9.699e-01
Iteracija 95/120 | proteklo 16.9 s | cilj 19.587471878 | Max|proj.grad| 9.196e-01
Iteracija 100/120 | proteklo 17.7 s | cilj 19.585237868 | Max|proj.grad| 2.116e+00
Iteracija 105/120 | proteklo 18.5 s | cilj 19.584209070 | Max|proj.grad| 4.121e-01
Iteracija 110/120 | proteklo 19.4 s | cilj 19.583700322 | Max|proj.grad| 8.556e-01
Iteracija 115/120 | proteklo 20.4 s | cilj 19.583139966 | Max|proj.grad| 1.163e+00
Iteracija 120/120 | proteklo 21.3 s | cilj 19.582513748 | Max|proj.grad| 4.723e-01
Zadržani cilj: 19.582513748
Max|proj.grad|: 4.723347e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.

loto7_4710_k86.csv: proveravam punu Qiskit istoriju i struje.

loto7_4710_k86.csv
NEXT: 1 x 4 y 15 z 33
Validacioni NLL: 20.017722094
Odvojeni test NLL: 19.818428365
Test NLL sa resetovanom memorijom: 22.571032082
Test NLL bez istorijskih ulaznih faza: 19.907169486
Modelna verovatnoća NEXT: 0.000111420965682
Normalizacioni odnos: 1.000000000000
Greška pune Qiskit istorije: 3.844e-15
Greška diskretnog zakona vođenja: 2.776e-17
Greška integrisanog transporta: 6.245e-17
Greška očuvanja mase transporta: 0.000e+00
Red kvadrature transporta: 16
Efekat memorije na NEXT margine: 4.299954e-01
Osetljivost poslednjih raspodela: 5.601686e-01
Zbir marginalnih verovatnoća: 7.000000000000
Završni Max|proj.grad|: 4.723347e-01
Test: broj kola prema broju pogodaka
0: 192 | 1: 387 | 2: 278 | 3: 72 | 4: 12 | 5: 1 | 6: 0 | 7: 0
Prosečan broj pogodaka na testu: 1.286624
Marginalne verovatnoće uključivanja brojeva:
01: 0.58030909 | 02: 0.02014429 | 03: 0.59056056 | 04: 0.34724328 | 05: 0.02333792 | 06: 0.17647855
07: 0.12525980 | 08: 0.00491043 | 09: 0.02335553 | 10: 0.39676858 | 11: 0.10064178 | 12: 0.11756979
13: 0.12628854 | 14: 0.14703191 | 15: 0.36787402 | 16: 0.26143419 | 17: 0.06162708 | 18: 0.08735982
19: 0.14413726 | 20: 0.15656802 | 21: 0.10721405 | 22: 0.25870471 | 23: 0.27240681 | 24: 0.28947276
25: 0.09797214 | 26: 0.09201824 | 27: 0.13621335 | 28: 0.04797237 | 29: 0.01721995 | 30: 0.21001908
31: 0.08454205 | 32: 0.17791754 | 33: 0.39667469 | 34: 0.10005570 | 35: 0.22018138 | 36: 0.21722283
37: 0.23062748 | 38: 0.06609538 | 39: 0.11856904
DIJAGNOSTIKA: 6 treninga nema potvrđen gradijentni uslov; njihovi minimumi nisu potvrđeni.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.03
Trening prelaza: 1785 | ridge: 0.03
Iteracija 5/120 | proteklo 0.8 s | cilj 19.745848768 | Max|proj.grad| 5.433e+00
Iteracija 10/120 | proteklo 1.2 s | cilj 19.695689851 | Max|proj.grad| 5.201e+00
Iteracija 15/120 | proteklo 1.6 s | cilj 19.681052902 | Max|proj.grad| 3.465e+00
Iteracija 20/120 | proteklo 1.9 s | cilj 19.674227880 | Max|proj.grad| 1.074e+00
Iteracija 25/120 | proteklo 2.3 s | cilj 19.663538048 | Max|proj.grad| 2.083e+00
Iteracija 30/120 | proteklo 2.6 s | cilj 19.653693888 | Max|proj.grad| 1.392e+00
Iteracija 35/120 | proteklo 2.9 s | cilj 19.648879766 | Max|proj.grad| 2.024e+00
Iteracija 40/120 | proteklo 3.8 s | cilj 19.630721283 | Max|proj.grad| 3.952e+00
Iteracija 45/120 | proteklo 4.2 s | cilj 19.616596875 | Max|proj.grad| 2.539e+00
Iteracija 50/120 | proteklo 4.5 s | cilj 19.612469361 | Max|proj.grad| 1.320e+00
Iteracija 55/120 | proteklo 4.8 s | cilj 19.608554166 | Max|proj.grad| 1.711e+00
Iteracija 60/120 | proteklo 5.2 s | cilj 19.601399937 | Max|proj.grad| 7.602e-01
Iteracija 65/120 | proteklo 5.6 s | cilj 19.597829731 | Max|proj.grad| 5.208e-01
Iteracija 70/120 | proteklo 6.0 s | cilj 19.593437471 | Max|proj.grad| 2.764e+00
Iteracija 75/120 | proteklo 6.4 s | cilj 19.585424754 | Max|proj.grad| 1.378e+00
Iteracija 80/120 | proteklo 6.7 s | cilj 19.582443841 | Max|proj.grad| 8.320e-01
Iteracija 85/120 | proteklo 7.2 s | cilj 19.576196038 | Max|proj.grad| 2.031e+00
Iteracija 90/120 | proteklo 7.5 s | cilj 19.567917572 | Max|proj.grad| 9.333e-01
Iteracija 95/120 | proteklo 7.8 s | cilj 19.566758779 | Max|proj.grad| 1.884e-01
Iteracija 100/120 | proteklo 8.4 s | cilj 19.564825953 | Max|proj.grad| 1.325e+00
Iteracija 105/120 | proteklo 8.7 s | cilj 19.562570352 | Max|proj.grad| 6.248e-01
Iteracija 110/120 | proteklo 9.0 s | cilj 19.561072671 | Max|proj.grad| 2.940e-01
Iteracija 115/120 | proteklo 9.3 s | cilj 19.559970800 | Max|proj.grad| 1.123e-01
Iteracija 120/120 | proteklo 9.6 s | cilj 19.559602095 | Max|proj.grad| 3.066e-01
Zadržani cilj: 19.559602095
Max|proj.grad|: 3.065842e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 20.220433168

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.03
Trening prelaza: 2083 | ridge: 0.03
Iteracija 5/120 | proteklo 1.1 s | cilj 19.639224873 | Max|proj.grad| 3.500e+00
Iteracija 10/120 | proteklo 1.4 s | cilj 19.630160376 | Max|proj.grad| 3.456e+00
Iteracija 15/120 | proteklo 1.9 s | cilj 19.617478781 | Max|proj.grad| 2.621e+00
Iteracija 20/120 | proteklo 2.4 s | cilj 19.605148263 | Max|proj.grad| 2.211e+00
Iteracija 25/120 | proteklo 2.9 s | cilj 19.594502209 | Max|proj.grad| 1.719e+00
Iteracija 30/120 | proteklo 3.2 s | cilj 19.592150221 | Max|proj.grad| 4.557e-01
Iteracija 35/120 | proteklo 3.8 s | cilj 19.591289593 | Max|proj.grad| 1.548e+00
Iteracija 40/120 | proteklo 4.2 s | cilj 19.588960184 | Max|proj.grad| 8.509e-01
Iteracija 45/120 | proteklo 4.7 s | cilj 19.583484989 | Max|proj.grad| 1.058e+00
Iteracija 50/120 | proteklo 5.2 s | cilj 19.580143337 | Max|proj.grad| 2.347e+00
Iteracija 55/120 | proteklo 5.5 s | cilj 19.576368789 | Max|proj.grad| 3.277e-01
Iteracija 60/120 | proteklo 5.9 s | cilj 19.575953379 | Max|proj.grad| 5.692e-01
Iteracija 65/120 | proteklo 6.3 s | cilj 19.575113928 | Max|proj.grad| 4.947e-01
Iteracija 70/120 | proteklo 6.7 s | cilj 19.574518742 | Max|proj.grad| 1.545e-01
Iteracija 75/120 | proteklo 7.1 s | cilj 19.573934319 | Max|proj.grad| 7.240e-01
Iteracija 80/120 | proteklo 7.4 s | cilj 19.572443346 | Max|proj.grad| 6.400e-01
Iteracija 85/120 | proteklo 8.0 s | cilj 19.569719754 | Max|proj.grad| 6.700e-01
Iteracija 90/120 | proteklo 8.4 s | cilj 19.567605002 | Max|proj.grad| 4.641e-01
Iteracija 95/120 | proteklo 8.8 s | cilj 19.565349877 | Max|proj.grad| 7.799e-01
Iteracija 100/120 | proteklo 9.1 s | cilj 19.563349158 | Max|proj.grad| 6.571e-01
Iteracija 105/120 | proteklo 9.5 s | cilj 19.561491435 | Max|proj.grad| 1.243e+00
Iteracija 110/120 | proteklo 9.9 s | cilj 19.558111531 | Max|proj.grad| 1.465e+00
Iteracija 115/120 | proteklo 10.3 s | cilj 19.554190258 | Max|proj.grad| 8.799e-01
Iteracija 120/120 | proteklo 10.8 s | cilj 19.551673954 | Max|proj.grad| 9.380e-01
Zadržani cilj: 19.551673954
Max|proj.grad|: 9.380278e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 20.064175980
Objedinjeni validacioni NLL za ridge 0.03: 20.142435883

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.3
Trening prelaza: 1785 | ridge: 0.3
Iteracija 5/120 | proteklo 1.0 s | cilj 19.767585517 | Max|proj.grad| 5.744e+00
Iteracija 10/120 | proteklo 1.4 s | cilj 19.738897400 | Max|proj.grad| 2.833e+00
Iteracija 15/120 | proteklo 1.7 s | cilj 19.731649401 | Max|proj.grad| 8.583e-01
Iteracija 20/120 | proteklo 2.1 s | cilj 19.726530888 | Max|proj.grad| 1.200e+00
Iteracija 25/120 | proteklo 2.5 s | cilj 19.722109799 | Max|proj.grad| 1.728e+00
Iteracija 30/120 | proteklo 2.8 s | cilj 19.714564521 | Max|proj.grad| 6.891e-01
Iteracija 35/120 | proteklo 3.2 s | cilj 19.688781854 | Max|proj.grad| 7.523e+00
Iteracija 40/120 | proteklo 3.7 s | cilj 19.659196427 | Max|proj.grad| 8.399e+00
Iteracija 45/120 | proteklo 4.1 s | cilj 19.634647353 | Max|proj.grad| 3.293e+01
Iteracija 50/120 | proteklo 4.7 s | cilj 19.612560253 | Max|proj.grad| 2.056e+00
Iteracija 55/120 | proteklo 5.0 s | cilj 19.597378387 | Max|proj.grad| 1.955e+00
Iteracija 60/120 | proteklo 5.6 s | cilj 19.577347288 | Max|proj.grad| 4.717e+00
Iteracija 65/120 | proteklo 6.0 s | cilj 19.532581436 | Max|proj.grad| 4.589e+00
Iteracija 70/120 | proteklo 6.3 s | cilj 19.517638687 | Max|proj.grad| 7.475e-01
Iteracija 75/120 | proteklo 6.8 s | cilj 19.504054241 | Max|proj.grad| 2.035e+00
Iteracija 80/120 | proteklo 7.5 s | cilj 19.487549395 | Max|proj.grad| 2.973e+00
Iteracija 85/120 | proteklo 8.1 s | cilj 19.467752621 | Max|proj.grad| 3.167e+00
Iteracija 90/120 | proteklo 8.5 s | cilj 19.454299023 | Max|proj.grad| 1.953e+00
Iteracija 95/120 | proteklo 9.0 s | cilj 19.443605122 | Max|proj.grad| 4.011e+00
Iteracija 100/120 | proteklo 9.4 s | cilj 19.433384580 | Max|proj.grad| 1.813e+00
Iteracija 105/120 | proteklo 9.8 s | cilj 19.426908210 | Max|proj.grad| 1.612e+00
Iteracija 110/120 | proteklo 10.1 s | cilj 19.421755887 | Max|proj.grad| 6.768e-01
Iteracija 115/120 | proteklo 10.5 s | cilj 19.420139336 | Max|proj.grad| 1.618e+00
Iteracija 120/120 | proteklo 10.9 s | cilj 19.412701975 | Max|proj.grad| 8.343e-01
Zadržani cilj: 19.412701975
Max|proj.grad|: 8.342692e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 19.894412193

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.3
Trening prelaza: 2083 | ridge: 0.3
Iteracija 5/120 | proteklo 0.9 s | cilj 19.476404141 | Max|proj.grad| 1.286e+00
Iteracija 10/120 | proteklo 1.3 s | cilj 19.474796952 | Max|proj.grad| 4.012e-01
Iteracija 15/120 | proteklo 1.7 s | cilj 19.469856113 | Max|proj.grad| 3.184e+00
Iteracija 20/120 | proteklo 2.1 s | cilj 19.458686022 | Max|proj.grad| 2.795e+00
Iteracija 25/120 | proteklo 2.6 s | cilj 19.442639267 | Max|proj.grad| 1.844e+00
Iteracija 30/120 | proteklo 3.0 s | cilj 19.430008192 | Max|proj.grad| 4.095e+00
Iteracija 35/120 | proteklo 3.4 s | cilj 19.425002864 | Max|proj.grad| 5.371e-01
Iteracija 40/120 | proteklo 3.9 s | cilj 19.422805816 | Max|proj.grad| 1.955e+00
Iteracija 45/120 | proteklo 4.4 s | cilj 19.419022116 | Max|proj.grad| 9.065e-01
Iteracija 50/120 | proteklo 4.7 s | cilj 19.417692816 | Max|proj.grad| 2.884e-01
Iteracija 55/120 | proteklo 5.3 s | cilj 19.413826289 | Max|proj.grad| 3.792e+00
Iteracija 60/120 | proteklo 5.7 s | cilj 19.405470988 | Max|proj.grad| 2.106e+00
Iteracija 65/120 | proteklo 6.1 s | cilj 19.400567504 | Max|proj.grad| 2.386e+00
Iteracija 70/120 | proteklo 6.7 s | cilj 19.396975782 | Max|proj.grad| 2.147e+00
Iteracija 75/120 | proteklo 7.1 s | cilj 19.393703959 | Max|proj.grad| 2.876e-01
Iteracija 80/120 | proteklo 7.4 s | cilj 19.392386756 | Max|proj.grad| 2.997e-01
Iteracija 85/120 | proteklo 7.9 s | cilj 19.390205318 | Max|proj.grad| 7.624e-01
Iteracija 90/120 | proteklo 8.3 s | cilj 19.388739747 | Max|proj.grad| 1.939e+00
Iteracija 95/120 | proteklo 8.7 s | cilj 19.384594392 | Max|proj.grad| 7.322e-01
Iteracija 100/120 | proteklo 9.2 s | cilj 19.383071383 | Max|proj.grad| 8.684e-01
Iteracija 105/120 | proteklo 9.6 s | cilj 19.381772257 | Max|proj.grad| 8.382e-01
Iteracija 110/120 | proteklo 10.0 s | cilj 19.379337338 | Max|proj.grad| 8.793e-01
Iteracija 115/120 | proteklo 10.3 s | cilj 19.377053991 | Max|proj.grad| 9.868e-01
Iteracija 120/120 | proteklo 10.7 s | cilj 19.374544209 | Max|proj.grad| 4.221e-01
Zadržani cilj: 19.374544209
Max|proj.grad|: 4.220558e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 20.215043268
Objedinjeni validacioni NLL za ridge 0.3: 20.054458293

loto7_4710_k86_loto_2977.csv: izabrani ridge 0.3
Validacioni NLL: 20.054458293

loto7_4710_k86_loto_2977.csv / model za odvojeni test
Trening prelaza: 2380 | ridge: 0.3
Iteracija 5/120 | proteklo 0.9 s | cilj 19.474995825 | Max|proj.grad| 2.590e+00
Iteracija 10/120 | proteklo 1.4 s | cilj 19.472661562 | Max|proj.grad| 2.169e+00
Iteracija 15/120 | proteklo 1.9 s | cilj 19.464524882 | Max|proj.grad| 1.901e+00
Iteracija 20/120 | proteklo 2.3 s | cilj 19.460517711 | Max|proj.grad| 7.671e-01
Iteracija 25/120 | proteklo 2.8 s | cilj 19.458491611 | Max|proj.grad| 6.631e-01
Iteracija 30/120 | proteklo 3.2 s | cilj 19.457027520 | Max|proj.grad| 1.220e+00
Iteracija 35/120 | proteklo 3.7 s | cilj 19.454358490 | Max|proj.grad| 8.489e-01
Iteracija 40/120 | proteklo 4.1 s | cilj 19.453519961 | Max|proj.grad| 4.368e-01
Iteracija 45/120 | proteklo 4.6 s | cilj 19.453294238 | Max|proj.grad| 1.071e-01
Iteracija 50/120 | proteklo 5.0 s | cilj 19.452780655 | Max|proj.grad| 1.449e-01
Iteracija 55/120 | proteklo 5.4 s | cilj 19.452675068 | Max|proj.grad| 2.528e-01
Iteracija 60/120 | proteklo 5.8 s | cilj 19.451802629 | Max|proj.grad| 2.122e-01
Iteracija 65/120 | proteklo 6.2 s | cilj 19.451583522 | Max|proj.grad| 1.743e-01
Iteracija 70/120 | proteklo 6.6 s | cilj 19.451110123 | Max|proj.grad| 4.729e-01
Iteracija 75/120 | proteklo 7.1 s | cilj 19.448979156 | Max|proj.grad| 6.370e-01
Iteracija 80/120 | proteklo 7.5 s | cilj 19.446626365 | Max|proj.grad| 6.936e-01
Iteracija 85/120 | proteklo 8.0 s | cilj 19.445220396 | Max|proj.grad| 3.648e-01
Iteracija 90/120 | proteklo 8.4 s | cilj 19.444119956 | Max|proj.grad| 5.282e-01
Iteracija 95/120 | proteklo 8.9 s | cilj 19.443355190 | Max|proj.grad| 7.450e-01
Iteracija 100/120 | proteklo 9.3 s | cilj 19.442259923 | Max|proj.grad| 1.132e+00
Iteracija 105/120 | proteklo 9.8 s | cilj 19.440717682 | Max|proj.grad| 9.820e-01
Iteracija 110/120 | proteklo 10.4 s | cilj 19.439227277 | Max|proj.grad| 2.651e-01
Iteracija 115/120 | proteklo 10.9 s | cilj 19.438119744 | Max|proj.grad| 4.461e-01
Iteracija 120/120 | proteklo 11.3 s | cilj 19.437608621 | Max|proj.grad| 3.035e-01
Zadržani cilj: 19.437608621
Max|proj.grad|: 3.035358e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv: odvojeni test je završen.
Test NLL: 19.981215096
Parametri tokom testa ostaju zamrznuti.
Svaki naredni test koristi samo tada poznata prethodna kola.
Kontrole nisu posebno trenirani konkurenti.

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima
Trening prelaza: 2976 | ridge: 0.3
Iteracija 5/120 | proteklo 1.2 s | cilj 19.532713007 | Max|proj.grad| 1.719e+00
Iteracija 10/120 | proteklo 1.7 s | cilj 19.530133058 | Max|proj.grad| 3.363e-01
Iteracija 15/120 | proteklo 2.2 s | cilj 19.529460691 | Max|proj.grad| 4.663e-01
Iteracija 20/120 | proteklo 2.7 s | cilj 19.527954796 | Max|proj.grad| 8.369e-01
Iteracija 25/120 | proteklo 3.2 s | cilj 19.524746949 | Max|proj.grad| 4.505e+00
Iteracija 30/120 | proteklo 3.9 s | cilj 19.520102047 | Max|proj.grad| 1.500e+00
Iteracija 35/120 | proteklo 4.6 s | cilj 19.516076126 | Max|proj.grad| 9.313e-01
Iteracija 40/120 | proteklo 5.3 s | cilj 19.512182556 | Max|proj.grad| 8.323e-01
Iteracija 45/120 | proteklo 5.8 s | cilj 19.510561601 | Max|proj.grad| 8.413e-01
Iteracija 50/120 | proteklo 6.4 s | cilj 19.508721450 | Max|proj.grad| 1.727e+00
Iteracija 55/120 | proteklo 6.9 s | cilj 19.506543342 | Max|proj.grad| 3.645e-01
Iteracija 60/120 | proteklo 7.4 s | cilj 19.505758337 | Max|proj.grad| 4.299e-01
Iteracija 65/120 | proteklo 8.0 s | cilj 19.505497868 | Max|proj.grad| 4.428e-01
Iteracija 70/120 | proteklo 8.5 s | cilj 19.504973871 | Max|proj.grad| 5.398e-01
Iteracija 75/120 | proteklo 9.0 s | cilj 19.504236685 | Max|proj.grad| 8.415e-01
Iteracija 80/120 | proteklo 9.5 s | cilj 19.503690826 | Max|proj.grad| 3.034e-01
Iteracija 85/120 | proteklo 10.2 s | cilj 19.502952213 | Max|proj.grad| 1.116e+00
Iteracija 90/120 | proteklo 10.8 s | cilj 19.500803042 | Max|proj.grad| 6.796e-01
Iteracija 95/120 | proteklo 11.4 s | cilj 19.499718431 | Max|proj.grad| 2.317e-01
Iteracija 100/120 | proteklo 11.9 s | cilj 19.499431316 | Max|proj.grad| 3.088e-01
Iteracija 105/120 | proteklo 12.4 s | cilj 19.499134778 | Max|proj.grad| 3.273e-01
Iteracija 110/120 | proteklo 12.9 s | cilj 19.498807463 | Max|proj.grad| 3.757e-01
Iteracija 115/120 | proteklo 13.4 s | cilj 19.498447987 | Max|proj.grad| 3.286e-01
Iteracija 120/120 | proteklo 14.0 s | cilj 19.498221821 | Max|proj.grad| 2.656e-01
Zadržani cilj: 19.498221821
Max|proj.grad|: 2.655802e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv: proveravam punu Qiskit istoriju i struje.

loto7_4710_k86_loto_2977.csv
NEXT: 2 x 12 y 18 z 34
Validacioni NLL: 20.054458293
Odvojeni test NLL: 19.981215096
Test NLL sa resetovanom memorijom: 22.563266823
Test NLL bez istorijskih ulaznih faza: 19.475335289
Modelna verovatnoća NEXT: 5.77656495605e-05
Normalizacioni odnos: 1.000000000000
Greška pune Qiskit istorije: 4.135e-15
Greška diskretnog zakona vođenja: 2.776e-17
Greška integrisanog transporta: 5.551e-17
Greška očuvanja mase transporta: 2.220e-16
Red kvadrature transporta: 16
Efekat memorije na NEXT margine: 5.980141e-01
Osetljivost poslednjih raspodela: 5.782824e-01
Zbir marginalnih verovatnoća: 7.000000000000
Završni Max|proj.grad|: 2.655802e-01
Test: broj kola prema broju pogodaka
0: 130 | 1: 244 | 2: 165 | 3: 49 | 4: 7 | 5: 1 | 6: 0 | 7: 0
Prosečan broj pogodaka na testu: 1.265101
Marginalne verovatnoće uključivanja brojeva:
01: 0.20859927 | 02: 0.35440075 | 03: 0.45278520 | 04: 0.28701600 | 05: 0.00396374 | 06: 0.28510781
07: 0.09674529 | 08: 0.29754407 | 09: 0.21042667 | 10: 0.07906803 | 11: 0.15941505 | 12: 0.30992694
13: 0.16024088 | 14: 0.12428950 | 15: 0.38563497 | 16: 0.19107185 | 17: 0.22044856 | 18: 0.35866215
19: 0.00354729 | 20: 0.43451482 | 21: 0.12981429 | 22: 0.22470325 | 23: 0.13355425 | 24: 0.24973149
25: 0.03458694 | 26: 0.16296850 | 27: 0.06949291 | 28: 0.17269020 | 29: 0.07269055 | 30: 0.01873289
31: 0.00735543 | 32: 0.03131744 | 33: 0.04053380 | 34: 0.45907533 | 35: 0.00799453 | 36: 0.03540695
37: 0.24818656 | 38: 0.02867069 | 39: 0.24908516
DIJAGNOSTIKA: 6 treninga nema potvrđen gradijentni uslov; njihovi minimumi nisu potvrđeni.
DIJAGNOSTIKA: Istorijske ulazne faze nisu dale niži test NLL od dijagnostičke kontrole bez tih faza.

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.03
Trening prelaza: 1039 | ridge: 0.03
Iteracija 5/120 | proteklo 0.7 s | cilj 19.732499487 | Max|proj.grad| 1.407e+01
Iteracija 10/120 | proteklo 1.1 s | cilj 19.644999297 | Max|proj.grad| 7.991e+00
Iteracija 15/120 | proteklo 1.5 s | cilj 19.604843168 | Max|proj.grad| 4.159e+00
Iteracija 20/120 | proteklo 1.7 s | cilj 19.568959233 | Max|proj.grad| 8.309e+00
Iteracija 25/120 | proteklo 2.0 s | cilj 19.528875201 | Max|proj.grad| 5.839e+00
Iteracija 30/120 | proteklo 2.3 s | cilj 19.507207635 | Max|proj.grad| 2.824e+00
Iteracija 35/120 | proteklo 2.6 s | cilj 19.493120536 | Max|proj.grad| 2.151e+00
Iteracija 40/120 | proteklo 2.8 s | cilj 19.477406738 | Max|proj.grad| 1.024e+00
Iteracija 45/120 | proteklo 3.0 s | cilj 19.461656731 | Max|proj.grad| 3.447e+00
Iteracija 50/120 | proteklo 3.3 s | cilj 19.436721023 | Max|proj.grad| 2.473e+00
Iteracija 55/120 | proteklo 3.6 s | cilj 19.397164758 | Max|proj.grad| 2.529e+00
Iteracija 60/120 | proteklo 3.8 s | cilj 19.380720258 | Max|proj.grad| 1.392e+00
Iteracija 65/120 | proteklo 4.0 s | cilj 19.375025349 | Max|proj.grad| 2.266e+00
Iteracija 70/120 | proteklo 4.3 s | cilj 19.361345094 | Max|proj.grad| 1.017e+00
Iteracija 75/120 | proteklo 4.5 s | cilj 19.352371756 | Max|proj.grad| 2.043e+00
Iteracija 80/120 | proteklo 4.8 s | cilj 19.342981332 | Max|proj.grad| 1.320e+00
Iteracija 85/120 | proteklo 5.0 s | cilj 19.339470721 | Max|proj.grad| 1.064e+00
Iteracija 90/120 | proteklo 5.2 s | cilj 19.333650067 | Max|proj.grad| 4.760e-01
Iteracija 95/120 | proteklo 5.4 s | cilj 19.329488143 | Max|proj.grad| 1.066e+00
Iteracija 100/120 | proteklo 5.6 s | cilj 19.325884922 | Max|proj.grad| 6.493e-01
Iteracija 105/120 | proteklo 5.8 s | cilj 19.324456204 | Max|proj.grad| 3.795e-01
Iteracija 110/120 | proteklo 6.0 s | cilj 19.320935223 | Max|proj.grad| 4.868e-01
Iteracija 115/120 | proteklo 6.2 s | cilj 19.318489796 | Max|proj.grad| 6.647e-01
Iteracija 120/120 | proteklo 6.4 s | cilj 19.316804849 | Max|proj.grad| 6.677e-01
Zadržani cilj: 19.316804849
Max|proj.grad|: 6.676922e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 19.876751655

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.03
Trening prelaza: 1212 | ridge: 0.03
Iteracija 5/120 | proteklo 0.5 s | cilj 19.363709199 | Max|proj.grad| 1.694e+00
Iteracija 10/120 | proteklo 0.8 s | cilj 19.357630678 | Max|proj.grad| 1.452e+00
Iteracija 15/120 | proteklo 1.0 s | cilj 19.351703295 | Max|proj.grad| 1.419e+00
Iteracija 20/120 | proteklo 1.2 s | cilj 19.346818379 | Max|proj.grad| 7.062e-01
Iteracija 25/120 | proteklo 1.4 s | cilj 19.344117729 | Max|proj.grad| 7.984e-01
Iteracija 30/120 | proteklo 1.7 s | cilj 19.339379895 | Max|proj.grad| 8.843e-01
Iteracija 35/120 | proteklo 1.9 s | cilj 19.337563249 | Max|proj.grad| 4.920e-01
Iteracija 40/120 | proteklo 2.2 s | cilj 19.335431968 | Max|proj.grad| 8.462e-01
Iteracija 45/120 | proteklo 2.5 s | cilj 19.329856686 | Max|proj.grad| 1.327e+00
Iteracija 50/120 | proteklo 2.7 s | cilj 19.327742521 | Max|proj.grad| 4.108e-01
Iteracija 55/120 | proteklo 3.0 s | cilj 19.326510650 | Max|proj.grad| 3.374e-01
Iteracija 60/120 | proteklo 3.2 s | cilj 19.324942100 | Max|proj.grad| 5.777e-01
Iteracija 65/120 | proteklo 3.4 s | cilj 19.323568589 | Max|proj.grad| 1.054e+00
Iteracija 70/120 | proteklo 3.6 s | cilj 19.322711875 | Max|proj.grad| 2.523e-01
Iteracija 75/120 | proteklo 3.9 s | cilj 19.320992859 | Max|proj.grad| 1.277e+00
Iteracija 80/120 | proteklo 4.1 s | cilj 19.317386072 | Max|proj.grad| 7.313e-01
Iteracija 85/120 | proteklo 4.3 s | cilj 19.315618066 | Max|proj.grad| 5.662e-01
Iteracija 90/120 | proteklo 4.6 s | cilj 19.313650675 | Max|proj.grad| 3.965e-01
Iteracija 95/120 | proteklo 4.8 s | cilj 19.313131023 | Max|proj.grad| 1.550e-01
Iteracija 100/120 | proteklo 5.0 s | cilj 19.312692122 | Max|proj.grad| 2.393e-01
Iteracija 105/120 | proteklo 5.2 s | cilj 19.312411617 | Max|proj.grad| 1.744e-01
Iteracija 110/120 | proteklo 5.5 s | cilj 19.312150342 | Max|proj.grad| 1.373e-01
Iteracija 115/120 | proteklo 5.7 s | cilj 19.311737140 | Max|proj.grad| 3.200e-01
Iteracija 120/120 | proteklo 6.0 s | cilj 19.311306398 | Max|proj.grad| 3.745e-01
Zadržani cilj: 19.311306398
Max|proj.grad|: 3.744901e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 19.945903180
Objedinjeni validacioni NLL za ridge 0.03: 19.911327417

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.3
Trening prelaza: 1039 | ridge: 0.3
Iteracija 5/120 | proteklo 0.5 s | cilj 19.717088425 | Max|proj.grad| 7.179e+00
Iteracija 10/120 | proteklo 0.9 s | cilj 19.640545000 | Max|proj.grad| 3.943e+00
Iteracija 15/120 | proteklo 1.2 s | cilj 19.583083608 | Max|proj.grad| 6.720e+00
Iteracija 20/120 | proteklo 1.5 s | cilj 19.549753026 | Max|proj.grad| 2.582e+00
Iteracija 25/120 | proteklo 1.8 s | cilj 19.532781886 | Max|proj.grad| 1.002e+00
Iteracija 30/120 | proteklo 2.1 s | cilj 19.500388702 | Max|proj.grad| 3.756e+00
Iteracija 35/120 | proteklo 2.3 s | cilj 19.472268523 | Max|proj.grad| 1.422e+00
Iteracija 40/120 | proteklo 2.6 s | cilj 19.458640150 | Max|proj.grad| 2.753e+00
Iteracija 45/120 | proteklo 2.8 s | cilj 19.442062751 | Max|proj.grad| 6.811e-01
Iteracija 50/120 | proteklo 3.0 s | cilj 19.436825582 | Max|proj.grad| 1.139e+00
Iteracija 55/120 | proteklo 3.3 s | cilj 19.427893832 | Max|proj.grad| 2.099e+00
Iteracija 60/120 | proteklo 3.5 s | cilj 19.416386701 | Max|proj.grad| 1.048e+00
Iteracija 65/120 | proteklo 3.7 s | cilj 19.412781157 | Max|proj.grad| 1.131e+00
Iteracija 70/120 | proteklo 3.9 s | cilj 19.406501551 | Max|proj.grad| 1.210e+00
Iteracija 75/120 | proteklo 4.1 s | cilj 19.400765960 | Max|proj.grad| 6.646e-01
Iteracija 80/120 | proteklo 4.3 s | cilj 19.396520760 | Max|proj.grad| 7.084e-01
Iteracija 85/120 | proteklo 4.5 s | cilj 19.392107827 | Max|proj.grad| 1.315e+00
Iteracija 90/120 | proteklo 4.8 s | cilj 19.386152295 | Max|proj.grad| 6.213e-01
Iteracija 95/120 | proteklo 5.0 s | cilj 19.380956248 | Max|proj.grad| 8.393e-01
Iteracija 100/120 | proteklo 5.2 s | cilj 19.373217243 | Max|proj.grad| 5.282e-01
Iteracija 105/120 | proteklo 5.4 s | cilj 19.370992496 | Max|proj.grad| 4.120e-01
Iteracija 110/120 | proteklo 5.6 s | cilj 19.367911673 | Max|proj.grad| 5.787e-01
Iteracija 115/120 | proteklo 5.8 s | cilj 19.363361073 | Max|proj.grad| 4.341e-01
Iteracija 120/120 | proteklo 6.0 s | cilj 19.362197840 | Max|proj.grad| 3.210e-01
Zadržani cilj: 19.362197840
Max|proj.grad|: 3.209752e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 20.108736633

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.3
Trening prelaza: 1212 | ridge: 0.3
Iteracija 5/120 | proteklo 0.5 s | cilj 19.453825663 | Max|proj.grad| 9.282e-01
Iteracija 10/120 | proteklo 0.8 s | cilj 19.444984803 | Max|proj.grad| 1.361e+00
Iteracija 15/120 | proteklo 1.0 s | cilj 19.431431232 | Max|proj.grad| 1.841e+00
Iteracija 20/120 | proteklo 1.2 s | cilj 19.425387204 | Max|proj.grad| 8.877e-01
Iteracija 25/120 | proteklo 1.5 s | cilj 19.422532922 | Max|proj.grad| 3.531e-01
Iteracija 30/120 | proteklo 1.7 s | cilj 19.420763638 | Max|proj.grad| 3.755e-01
Iteracija 35/120 | proteklo 1.9 s | cilj 19.417693653 | Max|proj.grad| 7.907e-01
Iteracija 40/120 | proteklo 2.2 s | cilj 19.415804880 | Max|proj.grad| 5.665e-01
Iteracija 45/120 | proteklo 2.4 s | cilj 19.413464747 | Max|proj.grad| 3.090e-01
Iteracija 50/120 | proteklo 2.6 s | cilj 19.411896535 | Max|proj.grad| 7.557e-01
Iteracija 55/120 | proteklo 2.8 s | cilj 19.410995908 | Max|proj.grad| 1.964e-01
Iteracija 60/120 | proteklo 3.1 s | cilj 19.409746851 | Max|proj.grad| 6.894e-01
Iteracija 65/120 | proteklo 3.3 s | cilj 19.407476186 | Max|proj.grad| 1.609e-01
Iteracija 70/120 | proteklo 3.5 s | cilj 19.406995948 | Max|proj.grad| 1.425e-01
Iteracija 75/120 | proteklo 3.8 s | cilj 19.405036289 | Max|proj.grad| 5.902e-01
Iteracija 80/120 | proteklo 4.0 s | cilj 19.403819006 | Max|proj.grad| 2.890e-01
Iteracija 85/120 | proteklo 4.3 s | cilj 19.402048751 | Max|proj.grad| 9.550e-01
Iteracija 90/120 | proteklo 4.5 s | cilj 19.397438018 | Max|proj.grad| 8.973e-01
Iteracija 95/120 | proteklo 4.7 s | cilj 19.394547520 | Max|proj.grad| 3.194e-01
Iteracija 100/120 | proteklo 5.0 s | cilj 19.393694191 | Max|proj.grad| 1.522e-01
Iteracija 105/120 | proteklo 5.2 s | cilj 19.393155517 | Max|proj.grad| 2.424e-01
Iteracija 110/120 | proteklo 5.4 s | cilj 19.392878066 | Max|proj.grad| 1.316e-01
Iteracija 115/120 | proteklo 5.6 s | cilj 19.392735784 | Max|proj.grad| 1.540e-01
Iteracija 120/120 | proteklo 5.9 s | cilj 19.392343317 | Max|proj.grad| 1.946e-01
Zadržani cilj: 19.392343317
Max|proj.grad|: 1.945657e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 19.684765049
Objedinjeni validacioni NLL za ridge 0.3: 19.896750841

loto7_4710_k86_loto_plus_1733.csv: izabrani ridge 0.3
Validacioni NLL: 19.896750841

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test
Trening prelaza: 1385 | ridge: 0.3
Iteracija 5/120 | proteklo 0.5 s | cilj 19.417962191 | Max|proj.grad| 8.585e-01
Iteracija 10/120 | proteklo 0.8 s | cilj 19.416772478 | Max|proj.grad| 4.857e-01
Iteracija 15/120 | proteklo 1.0 s | cilj 19.415633606 | Max|proj.grad| 5.179e-01
Iteracija 20/120 | proteklo 1.3 s | cilj 19.415035072 | Max|proj.grad| 2.060e-01
Iteracija 25/120 | proteklo 1.6 s | cilj 19.414783899 | Max|proj.grad| 1.594e-01
Iteracija 30/120 | proteklo 1.9 s | cilj 19.414520606 | Max|proj.grad| 6.072e-02
Iteracija 35/120 | proteklo 2.1 s | cilj 19.414288383 | Max|proj.grad| 1.894e-01
Iteracija 40/120 | proteklo 2.4 s | cilj 19.413806920 | Max|proj.grad| 1.628e-01
Iteracija 45/120 | proteklo 2.7 s | cilj 19.413413852 | Max|proj.grad| 2.259e-01
Iteracija 50/120 | proteklo 3.0 s | cilj 19.413079948 | Max|proj.grad| 1.959e-01
Iteracija 55/120 | proteklo 3.2 s | cilj 19.412528060 | Max|proj.grad| 1.589e-01
Iteracija 60/120 | proteklo 3.5 s | cilj 19.412195986 | Max|proj.grad| 1.426e-01
Iteracija 65/120 | proteklo 3.7 s | cilj 19.411897846 | Max|proj.grad| 8.013e-02
Iteracija 70/120 | proteklo 4.0 s | cilj 19.411788983 | Max|proj.grad| 6.922e-02
Iteracija 75/120 | proteklo 4.2 s | cilj 19.411665512 | Max|proj.grad| 9.489e-02
Iteracija 80/120 | proteklo 4.5 s | cilj 19.411631932 | Max|proj.grad| 3.361e-02
Iteracija 85/120 | proteklo 4.8 s | cilj 19.411595640 | Max|proj.grad| 3.685e-02
Iteracija 90/120 | proteklo 5.1 s | cilj 19.411394359 | Max|proj.grad| 1.514e-01
Iteracija 95/120 | proteklo 5.3 s | cilj 19.411223969 | Max|proj.grad| 2.123e-01
Iteracija 100/120 | proteklo 5.5 s | cilj 19.411028103 | Max|proj.grad| 4.769e-02
Iteracija 105/120 | proteklo 5.8 s | cilj 19.410926306 | Max|proj.grad| 5.147e-02
Iteracija 110/120 | proteklo 6.1 s | cilj 19.410858443 | Max|proj.grad| 4.351e-02
Iteracija 115/120 | proteklo 6.3 s | cilj 19.410833846 | Max|proj.grad| 1.650e-02
Iteracija 120/120 | proteklo 6.6 s | cilj 19.410804887 | Max|proj.grad| 1.117e-01
Zadržani cilj: 19.410804887
Max|proj.grad|: 1.116584e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv: odvojeni test je završen.
Test NLL: 20.112681584
Parametri tokom testa ostaju zamrznuti.
Svaki naredni test koristi samo tada poznata prethodna kola.
Kontrole nisu posebno trenirani konkurenti.

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima
Trening prelaza: 1732 | ridge: 0.3
Iteracija 5/120 | proteklo 0.7 s | cilj 19.525062783 | Max|proj.grad| 8.149e-01
Iteracija 10/120 | proteklo 1.0 s | cilj 19.518949394 | Max|proj.grad| 8.241e-01
Iteracija 15/120 | proteklo 1.5 s | cilj 19.508722931 | Max|proj.grad| 2.508e+00
Iteracija 20/120 | proteklo 1.9 s | cilj 19.499361207 | Max|proj.grad| 1.319e+00
Iteracija 25/120 | proteklo 2.2 s | cilj 19.491122826 | Max|proj.grad| 2.595e+00
Iteracija 30/120 | proteklo 2.5 s | cilj 19.481706196 | Max|proj.grad| 2.249e+00
Iteracija 35/120 | proteklo 3.0 s | cilj 19.478363957 | Max|proj.grad| 2.444e+00
Iteracija 40/120 | proteklo 3.4 s | cilj 19.473509914 | Max|proj.grad| 1.245e+00
Iteracija 45/120 | proteklo 3.7 s | cilj 19.470089741 | Max|proj.grad| 1.561e+00
Iteracija 50/120 | proteklo 4.2 s | cilj 19.462049395 | Max|proj.grad| 2.196e+00
Iteracija 55/120 | proteklo 4.5 s | cilj 19.455805340 | Max|proj.grad| 9.470e-01
Iteracija 60/120 | proteklo 4.8 s | cilj 19.451497727 | Max|proj.grad| 8.313e-01
Iteracija 65/120 | proteklo 5.2 s | cilj 19.448076045 | Max|proj.grad| 4.629e-01
Iteracija 70/120 | proteklo 5.6 s | cilj 19.446827560 | Max|proj.grad| 1.710e+00
Iteracija 75/120 | proteklo 5.9 s | cilj 19.441670810 | Max|proj.grad| 1.784e+00
Iteracija 80/120 | proteklo 6.2 s | cilj 19.438407515 | Max|proj.grad| 7.908e-01
Iteracija 85/120 | proteklo 6.6 s | cilj 19.436263837 | Max|proj.grad| 4.517e-01
Iteracija 90/120 | proteklo 6.9 s | cilj 19.434985580 | Max|proj.grad| 6.139e-01
Iteracija 95/120 | proteklo 7.2 s | cilj 19.433518351 | Max|proj.grad| 2.261e-01
Iteracija 100/120 | proteklo 7.6 s | cilj 19.431559753 | Max|proj.grad| 1.558e+00
Iteracija 105/120 | proteklo 8.0 s | cilj 19.428067519 | Max|proj.grad| 7.322e-01
Iteracija 110/120 | proteklo 8.3 s | cilj 19.425800559 | Max|proj.grad| 7.318e-01
Iteracija 115/120 | proteklo 8.6 s | cilj 19.422936445 | Max|proj.grad| 9.589e-01
Iteracija 120/120 | proteklo 8.9 s | cilj 19.420294272 | Max|proj.grad| 4.284e-01
Zadržani cilj: 19.420294272
Max|proj.grad|: 4.283547e-01
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Zadržani kandidat ima validnu raspodelu; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv: proveravam punu Qiskit istoriju i struje.

loto7_4710_k86_loto_plus_1733.csv
NEXT: 8 x 14 y 19 z 29
Validacioni NLL: 19.896750841
Odvojeni test NLL: 20.112681584
Test NLL sa resetovanom memorijom: 22.222123528
Test NLL bez istorijskih ulaznih faza: 19.467519528
Modelna verovatnoća NEXT: 7.97556695693e-05
Normalizacioni odnos: 1.000000000000
Greška pune Qiskit istorije: 2.444e-15
Greška diskretnog zakona vođenja: 1.821e-17
Greška integrisanog transporta: 5.204e-17
Greška očuvanja mase transporta: 1.110e-16
Red kvadrature transporta: 16
Efekat memorije na NEXT margine: 6.868270e-01
Osetljivost poslednjih raspodela: 5.444765e-01
Zbir marginalnih verovatnoća: 7.000000000000
Završni Max|proj.grad|: 4.283547e-01
Test: broj kola prema broju pogodaka
0: 73 | 1: 145 | 2: 103 | 3: 18 | 4: 8 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka na testu: 1.259366
Marginalne verovatnoće uključivanja brojeva:
01: 0.11871477 | 02: 0.02368261 | 03: 0.03601434 | 04: 0.13769069 | 05: 0.02370418 | 06: 0.12007539
07: 0.26862887 | 08: 0.49311577 | 09: 0.08361686 | 10: 0.37177938 | 11: 0.09430517 | 12: 0.05134896
13: 0.19609617 | 14: 0.39112597 | 15: 0.05356494 | 16: 0.27388640 | 17: 0.24257026 | 18: 0.39469088
19: 0.39154437 | 20: 0.04168102 | 21: 0.26473521 | 22: 0.24586569 | 23: 0.06559978 | 24: 0.03422836
25: 0.10535693 | 26: 0.34305254 | 27: 0.23782409 | 28: 0.29560605 | 29: 0.45744446 | 30: 0.18363022
31: 0.02745328 | 32: 0.08382385 | 33: 0.09345312 | 34: 0.03425541 | 35: 0.11183232 | 36: 0.30639695
37: 0.05871946 | 38: 0.01185682 | 39: 0.23102844
DIJAGNOSTIKA: 6 treninga nema potvrđen gradijentni uslov; njihovi minimumi nisu potvrđeni.
DIJAGNOSTIKA: Istorijske ulazne faze nisu dale niži test NLL od dijagnostičke kontrole bez tih faza.

ZAVRŠNI NEXT REZULTATI
loto7_4710_k86.csv | NEXT: 1 x 4 y 15 z 33
loto7_4710_k86_loto_2977.csv | NEXT: 2 x 12 y 18 z 34
loto7_4710_k86_loto_plus_1733.csv | NEXT: 8 x 14 y 19 z 29
"""



"""
Ideja	                        Šta je korisno za moj Loto 7/39
A Primer on Quantum Mechanics	Kvantna memorija: stanje koje se prenosi između kola, umesto ponovnog pripremanja nezavisnog stanja za svaki uzorak. 
Bohm i pilot-talasi	            Razlikovanje amplitude, faze i struje verovatnoće. Istražiti kako se raspodela menja kroz istorijske prelaze, uz očuvanje ukupne verovatnoće.
The Physics of Consciousness	Promenljiva mreža odnosa može biti inspiracija. Međutim, članak opisuje Cahillovu spekulativnu teoriju sa suštinskom ulogom slučajnog šuma. 


Za Bohmovu oblast sledeća konstrukcija:
- Qiskit stanje nosi kompleksne amplitude i pamti prethodna kola.
- Učeni Hamiltonijan određuje evoluciju tog stanja.
- Struje verovatnoće pokazuju kako fazni odnosi prenose verovatnoću između stanja.
- Istorijski prelazi određuju parametre evolucije; završna raspodela daje NEXT.

Dinamika stanja kroz vreme, uz proveru očuvanja verovatnoće. 
Brojevi 1-39 predstavljali bi oznake stanja; ne bi pretpostavili da su susedni brojevi fizički blizu.

Za doslovnu pilot-talasnu konstrukciju potrebno je definisati i zakon vođenja konfiguracije. 
Sama kompleksna amplituda nije dovoljna da model nazovem Bohmovim. 
Bohmov originalni rad takođe ne daje dodatnu prediktivnu moć samom promenom interpretacije.
Bohm — evolucija amplituda, faza i struja verovatnoće u Qiskitu.

Ovo je diskretna adaptacija inspirisana Bohmom: 
kvantna memorija, učeni Hamiltonijan i struje verovatnoće. 
Zakon transporta koristi diskretnu formulaciju vođenja raspodele; ne uzorkuje putanje čestica. 
Osnova diskretne formulacije.
"""
