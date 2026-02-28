import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import Aer 
from qiskit import *
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter 

from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from scipy.linalg import expm

from qiskit.circuit.library import zz_feature_map, RealAmplitudes


simulator = Aer.get_backend('aer_simulator')

# QVC Circuit
def circuit(phi, theta, params):
   # measurement = ClassicalRegister(1, 'measurement')
   # circ = QuantumCircuit(2, 1)
    clock = QuantumRegister(2, 'clock')
    measurement = ClassicalRegister(1, 'measurement')
    circ = QuantumCircuit(clock, measurement)
    # state preparation
    circ.ry(phi,0)
    circ.ry(theta,1)

    # layer 1
    theta1 = Parameter('theta1')
    phi1 = Parameter('phi1')
    circ.rx(theta1,0)
    circ.rz(phi1,0)
    theta2 = Parameter('theta2')
    phi2 = Parameter('phi2')
    circ.rx(theta2,1)
    circ.rz(phi2,1)
    circ.cx(0,1)
    circ.cx(1,0)

    # layer 2
    theta3 = Parameter('theta3')
    phi3 = Parameter('phi3')
    circ.rx(theta3,0)
    circ.rz(phi3,0)
    theta4 = Parameter('theta4')
    phi4 = Parameter('phi4')
    circ.rx(theta4,1)
    circ.rz(phi4,1)
    circ.cx(0,1)
    circ.cx(1,0)

    # layer 1
    theta5 = Parameter('theta5')
    phi5 = Parameter('phi5')
    circ.rx(theta5,0)
    circ.rz(phi5,0)
    theta6 = Parameter('theta6')
    phi6 = Parameter('phi6')
    circ.rx(theta6,1)
    circ.rz(phi6,1)
    circ.cx(0,1)
    circ.cx(1,0)

    
    bound_circuit = circ.assign_parameters({theta1 : params["theta1"]
                                           , phi1 : params["phi1"]
                                           , theta2 : params["theta2"]
                                           , phi2 : params["phi2"] 
                                           , theta3 : params["theta3"]
                                           , phi3 : params["phi3"]
                                           , theta4 : params["theta4"]
                                           , phi4 : params["phi4"] 
                                           , theta5 : params["theta5"]
                                           , phi5 : params["phi5"]
                                           , theta6 : params["theta6"]
                                           , phi6 : params["phi6"] } )
    bound_circuit.measure(0,measurement)
   # bound_circuit.draw('mpl',fold=10, filename="QVC circuit.png")
   
    return bound_circuit

def calculateDeriv(phi, theta, paramsp, paramsm):
    circp = circuit(phi, theta, paramsp)
    qc_compiled = transpile(circp, simulator)
    result = simulator.run(qc_compiled, shots=1024).result()
    countsp = result.get_counts(qc_compiled)  
    if '1' not in countsp: evp = 1
    elif '0' not in countsp: evp = -1
    else: evp = (countsp['0'] - countsp['1'])/1024

    circm = circuit(phi, theta, paramsm)
    qc_compiled = transpile(circm, simulator)
    result = simulator.run(qc_compiled, shots=1024).result()
    countsm = result.get_counts(qc_compiled)
    if '1' not in countsm: evm = 1
    elif '0' not in countsm: evm = -1
    else: evm = (countsm['0'] - countsm['1'])/1024
    return 0.5 * (evp - evm)

def calculatePartialDerivTheta1(phi, theta, params):
    paramsp = dict(params)
    paramsp["theta1"] = paramsp["theta1"] + np.pi/2

    paramsm = dict(params)
    paramsm["theta1"] = paramsm["theta1"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivPhi1(phi, theta, params):
    paramsp = dict(params)
    paramsp["phi1"] = paramsp["phi1"] + np.pi/2

    paramsm = dict(params)
    paramsm["phi1"] = paramsm["phi1"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivTheta2(phi, theta, params):
    paramsp = dict(params)
    paramsp["theta2"] = paramsp["theta2"] + np.pi/2

    paramsm = dict(params)
    paramsm["theta2"] = paramsm["theta2"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivPhi2(phi, theta, params):
    paramsp = dict(params)
    paramsp["phi2"] = paramsp["phi2"] + np.pi/2

    paramsm = dict(params)
    paramsm["phi2"] = paramsm["phi2"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivTheta3(phi, theta, params):
    paramsp = dict(params)
    paramsp["theta3"] = paramsp["theta3"] + np.pi/2

    paramsm = dict(params)
    paramsm["theta3"] = paramsm["theta3"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivPhi3(phi, theta, params):
    paramsp = dict(params)
    paramsp["phi3"] = paramsp["phi3"] + np.pi/2

    paramsm = dict(params)
    paramsm["phi3"] = paramsm["phi3"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivTheta4(phi, theta, params):
    paramsp = dict(params)
    paramsp["theta4"] = paramsp["theta4"] + np.pi/2

    paramsm = dict(params)
    paramsm["theta4"] = paramsm["theta4"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivPhi4(phi, theta, params):
    paramsp = dict(params)
    paramsp["phi4"] = paramsp["phi4"] + np.pi/2

    paramsm = dict(params)
    paramsm["phi4"] = paramsm["phi4"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivTheta5(phi, theta, params):
    paramsp = dict(params)
    paramsp["theta5"] = paramsp["theta5"] + np.pi/2

    paramsm = dict(params)
    paramsm["theta5"] = paramsm["theta5"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivPhi5(phi, theta, params):
    paramsp = dict(params)
    paramsp["phi5"] = paramsp["phi5"] + np.pi/2

    paramsm = dict(params)
    paramsm["phi5"] = paramsm["phi5"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivTheta6(phi, theta, params):
    paramsp = dict(params)
    paramsp["theta6"] = paramsp["theta6"] + np.pi/2

    paramsm = dict(params)
    paramsm["theta6"] = paramsm["theta6"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)

def calculatePartialDerivPhi6(phi, theta, params):
    paramsp = dict(params)
    paramsp["phi6"] = paramsp["phi6"] + np.pi/2

    paramsm = dict(params)
    paramsm["phi6"] = paramsm["phi6"] - np.pi/2

    return calculateDeriv(phi, theta, paramsp, paramsm)