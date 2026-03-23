import numpy as np

def paulix():
    return np.array([[0,1],[1,0]])

def pauliy():
    return 1j*np.array([[0,-1],[1,0]])

def pauliz():
    return np.array([[1,0],[0,-1]])