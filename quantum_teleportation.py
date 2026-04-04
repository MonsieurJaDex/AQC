import numpy as np

import pennylane as qml

from pennylane.measurements import ExpectationMP

from concurrent.futures import ProcessPoolExecutor, as_completed

import json

# set qubit state from angle matrix
def init_qubit(matrix: np.ndarray, qubit: int):
    qml.RX(phi=matrix[0], wires=qubit)
    qml.RY(phi=matrix[1], wires=qubit)
    qml.RZ(phi=matrix[2], wires=qubit)

# get qubit state as matrix
def get_qubit_state(qubit: int) -> tuple[ExpectationMP, ExpectationMP, ExpectationMP]:
    return qml.expval(qml.PauliX(wires=qubit)), qml.expval(qml.PauliY(wires=qubit)), qml.expval(qml.PauliZ(wires=qubit))

INITIAL_QBIT = 0
ALICE_QBIT = 1
BOB_QBIT = 9

def Hadamard(wires):
    qml.RZ(np.pi, wires=wires)
    qml.RY(np.pi/2, wires=wires)
    qml.RZ(np.pi, wires=wires)

def MS(wires: list):
    qml.IsingXX(np.pi/2, wires=[wires[0], wires[1]])

# Google SD-style
def CNOT(wires: list):
    _, target = wires
    Hadamard(target)
    qml.CZ(wires=wires)
    qml.Hadamard(target)

def CNOT_ionq(wires: list):
    control, target = wires
    qml.RY(np.pi/2, wires=control)
    MS(wires)
    qml.RX(-np.pi/2, wires=control)
    qml.RX(-np.pi/2, wires=target)
    qml.RY(-np.pi/2, wires=control)

def SWAP(wires: list):
    CNOT(wires)
    CNOT(wires[::-1])
    CNOT(wires)


matrix = np.array([
    np.pi, np.pi/4, 0
])


def run_ionq(matrix):
    device = qml.device("qiskit.aer", wires=3)

    @qml.qnode(device)
    def circuit(matrix: np.ndarray):
        init_qubit(matrix, INITIAL_QBIT)
        
        qml.Hadamard(ALICE_QBIT)
        CNOT_ionq([ALICE_QBIT, BOB_QBIT])
        
        CNOT_ionq([INITIAL_QBIT, ALICE_QBIT])
        qml.Hadamard(INITIAL_QBIT)
    
        m0 = qml.measure(INITIAL_QBIT)
        m1 = qml.measure(ALICE_QBIT)
        
        qml.cond(m1, qml.PauliX)(wires=BOB_QBIT)
        qml.cond(m0, qml.PauliZ)(wires=BOB_QBIT)
        
        return get_qubit_state(BOB_QBIT)
    
    return circuit(matrix)


def run_sd(matrix):
    device = qml.device("qiskit.aer", wires=10)

    @qml.qnode(device)
    def circuit(matrix: np.ndarray):
        init_qubit(matrix, INITIAL_QBIT)

        new_bob_qbit = BOB_QBIT

        # SWAP BOB_QBIT to ALICE_QBIT+1 position
        if new_bob_qbit - ALICE_QBIT != 1:
            while(new_bob_qbit != ALICE_QBIT + 1):
                SWAP([new_bob_qbit, new_bob_qbit-1])
                new_bob_qbit = new_bob_qbit - 1
                


        qml.Hadamard(ALICE_QBIT)
        CNOT([ALICE_QBIT, new_bob_qbit])
        
        CNOT([INITIAL_QBIT, ALICE_QBIT])
        qml.Hadamard(INITIAL_QBIT)

        m0 = qml.measure(INITIAL_QBIT)
        m1 = qml.measure(ALICE_QBIT)
        
        qml.cond(m1, qml.PauliX)(wires=new_bob_qbit)
        qml.cond(m0, qml.PauliZ)(wires=new_bob_qbit)
        
        return get_qubit_state(new_bob_qbit)

    return circuit(matrix)

def run_abstract(matrix):
    device = qml.device("default.mixed", wires=10)

    @qml.qnode(device)
    def circuit(matrix: np.ndarray):
        init_qubit(matrix, INITIAL_QBIT)

        qml.Hadamard(wires=ALICE_QBIT)
        qml.CNOT(wires=[ALICE_QBIT, BOB_QBIT])
        qml.CNOT(wires=[INITIAL_QBIT, ALICE_QBIT])
        qml.Hadamard(wires=INITIAL_QBIT)

        # measure 1 and 2 qubits
        m0 = qml.measure(INITIAL_QBIT)
        m1 = qml.measure(ALICE_QBIT)
        
        qml.cond(m1, qml.PauliX)(wires=BOB_QBIT) # type: ignore
        qml.cond(m0, qml.PauliZ)(wires=BOB_QBIT) # type: ignore

        return get_qubit_state(BOB_QBIT)
    
    return circuit(matrix)

if __name__ == "__main__":    
    num_simulations = 10000
    process = 15
    results = []

    print(f"Запускаем {num_simulations} симуляций с {process} процессами...")

    with ProcessPoolExecutor(max_workers=process) as executor:
        futures = [executor.submit(run_sd, matrix) for _ in range(num_simulations)]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                print(f"Готово: {len(results)}/{num_simulations}", end="\r")
            except Exception as e:
                print(f"\nОшибка в процессе: {type(e).__name__}: {e}")

    results.append(run_abstract(matrix))

    print(f"\nВсе симуляции завершены. Успешно: {len(results)}")

    if results:
        results = np.array(results)
        np.save("sim_sd_10q.npy", results)
    else:
        print("Нет успешных результатов для построения графика")