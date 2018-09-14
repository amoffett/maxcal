import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument('--msm_file', type=str, help='Name of the pickle file containing the original MSM (MSMBuilder Markov state model object).')
parser.add_argument('--mu_file', type=str, help='Name of the pickle file containing a dictionary with the transfer free energy (kcal/mol) from transfer_free_energy.py.')
parser.add_argument('--temperature', type=float, help='Temperature (Kelvin).')
parser.add_argument('--out_file', type=str, help='Prefix for the name of the output sparse Market Matrix file.')
args = parser.parse_args()

import numpy as np
import pickle as pkl
from scipy import constants
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

epsilon = 1e-15
R = (1/constants.calorie)/1000.0 * constants.R
beta = 1 / (R * args.temperature)

msm = pkl.load(open(args.msm_file,"rb"))
mu = pkl.load(open(args.mu_file,"rb"))
 
T0 = msm.transmat_.T
pi0 = msm.populations_

if pi0.shape[0] != len(mu.keys()):
        raise ValueError('Mismatch between number of MSM states and number of transfer free energies.')

pi_star = (pi0 * np.exp(-beta * np.array(mu.values()))) / np.sum(pi0 * np.exp(-beta * np.array(mu.values())))
w0 = np.ones([pi0.shape[0]]).astype(float)

T1 = T0 * np.sqrt((pi_star*T0*w0).T/(pi_star*T0*w0)) * w0
w = w0 / T1.sum(1)
T = [T0,T1]
while np.any(np.abs(T[-1] - T[-2]) > epsilon):
        T_star = T0 * np.sqrt((pi_star*T0*w).T/(pi_star*T0*w)) * w
        w = w / T_star.sum(axis=0)
        T.append(T_star)

T_star_sparse = csr_matrix(T[-1].T)
mmwrite(args.out_file + '.mtx',T_star_sparse)
