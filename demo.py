# This is a demo script to calculate the energy of a given molecule using QCC-CAS. The gates and parameters are directly read from the data files.

import os
import sys
from typing import Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo
from pyscf.tools import mo_mapping


from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit_nature import settings
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering

settings.use_pauli_sum_op = False

def get_hamiltoniam(path: str, nelec: int, mapper: Union[JordanWignerMapper, ParityMapper]) -> SparsePauliOp:
    """
    Read molecular data from .npz file and return mapped qubit Hamiltonian.

    Args:
        path: Path to the .npz file containing h1 and h2 integrals
        nelec: Total number of electrons
        mapper: Fermion-to-qubit mapping operator

    Returns:
        Mapped qubit Hamiltonian as SparsePauliOp
    """
    
    ham = np.load(path + '.npz')
    h1 = ham['h1_ao']
    h2 = ham['h2_ao']
    fakemol = gto.M(verbose=0)
    norb = h1.shape[0]
    fakemol.nelectron = nelec
    fake_hf = scf.RHF(fakemol)
    fake_hf._eri = ao2mo.restore(1, h2, norb)
    fake_hf.get_hcore = lambda *args: h1
    fake_hf.get_ovlp = lambda *args: np.eye(norb)
    fake_hf.kernel()
    mo = fake_hf.mo_coeff
    h1_mo = np.einsum('ji,jk,kl->il', mo, h1, mo)
    h2_mo = ao2mo.incore.general(fake_hf._eri,(mo, mo, mo, mo), compact = False)
    hamiltonian = ElectronicEnergy.from_raw_integrals(h1_mo, to_physicist_ordering(h2_mo))

    hamiltonian_q = mapper.map(hamiltonian.second_q_op())
    return hamiltonian_q


def calc_energy_qcc(norb: int, ne: tuple, mapper: Union[JordanWignerMapper, ParityMapper], 
                      paras: List[float], op: SparsePauliOp, paulis: List[SparsePauliOp]) -> float:
    """
    Calculate the energy expectation value using the QCC ansatz.

    Args:
        norb: Number of orbitals
        ne: Tuple of (alpha, beta) electrons
        mapper: Fermion-to-qubit mapping operator
        paras: List of ansatz parameters
        op: Qubit Hamiltonian
        paulis: List of Pauli string generators

    Returns:
        Energy expectation value
    """

    hf_state = HartreeFock(norb, ne, mapper)
    state = Statevector(hf_state)
    qc = QuantumCircuit(state.num_qubits)            
    for pg, para in zip(paulis[::-1], paras[::-1]):
        peg = PauliEvolutionGate(pg, time = para/2)
        qc.append(peg, range(state.num_qubits))
    
    wf_new = state.evolve(qc)
    energy = wf_new.expectation_value(op).real
    return energy

if __name__ == "__main__":
    # Configuration
    cas = 4 # Can be 2, 4, 6
    mol = 'o3' # Can be 'li4', 'o3'
    ne = (cas//2, cas//2)
    norb = sum(ne)
    mapper = ParityMapper(num_particles=ne)
    
    # Load reference data
    data_ref = pd.read_csv(f'./data/energy_cas{cas}_{mol}.csv')
    e_ref = data_ref['QCC-CAS'] - data_ref['ecore']
    
    e_calc = []

    # Verify energies at different bond lengths
    for d in np.arange(-0.1, 0.4, 0.05):
        name_ham = f'./data/hamiltonian_cas{cas}_{mol}_{d:2.2f}'
        hamiltonian_q = get_hamiltoniam(name_ham, sum(ne), mapper=mapper)

        data_qcc = pd.read_csv(f'./data/qcc_cas{cas}_{mol}_{d:2.2f}.csv')
        ansatz_paras = np.array(data_qcc['parameters']).tolist()
        ansatz_gate = [SparsePauliOp(p) for p in data_qcc['pauligate']]

        qcc_energy = calc_energy_qcc(norb, ne, mapper, ansatz_paras, hamiltonian_q, ansatz_gate)
        e_calc.append(qcc_energy)
        e_old = e_ref[data_ref['bond length'] == np.round(d, 2)].to_numpy()[0]
        assert np.abs(qcc_energy - e_old) < 1e-5, f'QCC-CAS - old: {qcc_energy-e_old:.5f}'
    print('Finished all QCC-CAS energy calculation and verification passed.')
    print('QCC-CAS energy: ', e_calc)