import numpy as np
import scipy
import scipy.io
import torch
import pdb

from mypypower.newtonpf import newtonpf
from pypower.ppoption import ppoption

Zbase = 1;
Vbase = 4800;
Sbase = Vbase **2 / Zbase
'''
def getSbus(P, Q, fac = Sbase/1000):
    ## This expects P and Q in kW
    P = P/fac;
    Q = Q/fac;
    return P + 1j*Q
'''
class IEEE37():
    def __init__(self, filePath = './network/IEEE-37',
                       dataPath = './data'):
        Ybus = scipy.io.loadmat(f'{filePath}/Ybus.mat')
        self.Ybus = Ybus['Ybus']
        self.n = self.Ybus.shape[0]
        self.v_lower = 0.95
        self.v_upper = 1.05
        
        # Load linearized model
        R = scipy.io.loadmat(f'{filePath}_linearized/R.mat')
        B = scipy.io.loadmat(f'{filePath}_linearized/B.mat')
        self.R = R['R']
        self.B = B['B']
        
        ## Bus index lists of each type of bus
        self.ref = np.array([0])
        self.pv = np.array([], dtype = np.int32) #np.array([4, 7, 9, 10, 11, 13, 16, 17, 20, 22, 23, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36])-1
        self.pq = np.array([i  for i in range(self.n) if (i not in self.ref) & (i not in self.pv)], dtype = np.int32)
        self.n_pq = len(self.pq)
        self.ppopt = ppoption()
        
        self._get_reference()
        self._get_load_and_gen(dataPath = dataPath)
        
    def getSbus(self, t, wrt_reference = True, w_slack = False):
        '''
            Returns the vector of complex bus power injections, that is, generation
            minus load. Power is expressed in per unit.
        '''
        P = self.P_gen[t] - self.P_l[t]
        Q = - self.Q_l[t]
        S = P + 1j*Q
        P_av = self.P_gen[t]
        
        if wrt_reference:
            S = S - self.S0
        
        if w_slack:
            return S, P_av[self.gen_idx]
        else:
            return S[-self.n_pq:], P_av[self.gen_idx]

    def step(self, Sbus, wrt_reference = True):
        '''
        returns:
            voltage magitude, solver flag
        '''
        if wrt_reference:
            S = self.S0.copy()
            S[-len(Sbus):] += Sbus
        else:
            S = Sbus
        V, success, _  = newtonpf(scipy.sparse.csr_matrix(self.Ybus), S, self.V0, self.ref, self.pv, self.pq, self.ppopt)
        return np.abs(V), success
    
    def linear_estimate(self, P, Q, wrt_reference = True):
        if wrt_reference:
            if torch.is_tensor(P):
                return torch.tensor(self.R).float().matmul(P) + torch.tensor(self.B).float().matmul(Q)
            else:
                return self.R.dot(P) + self.B.dot(Q)
        else:
            V = self.V0.copy()
            delta_p = P-self.P0
            delta_q = Q-self.Q0
            V[-self.n_pq:] += self.R.dot(delta_p[-self.n_pq:]) + self.B.dot(delta_q[-self.n_pq:])
            return V
    
    ## Reference Point for Linearization
    def _get_reference(self):
        # Flat voltage point
        self.V0 = np.ones(self.n);
        A0 = np.zeros(self.n);
        # Corresponding to current injection
        J0 = self.Ybus.dot(self.V0*np.exp(1j*A0));
        # Corresponding to power injection
        S0 = self.V0*np.exp(1j*A0)*np.conj(J0);
        self.P0 = np.real(S0);
        self.Q0 = np.imag(S0);
        self.S0 = self.P0 + 1j*self.Q0
        
    ## Load Demand and Generation
    def _get_load_and_gen(self, dataPath = './data'):
        # Load
        self.load_idx = np.array([2, 5, 6, 7, 9, 10, 11, 13, 14, 16, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 30, 32, 33, 35, 36]) -1
        load = scipy.io.loadmat(f'{dataPath}/Loads_1sec.mat') #(Unit in W)
        load = load['Loads'].transpose() # 604800 x 8
        self.P_l = np.zeros((load.shape[0], self.n))
        for i, idx in enumerate(self.load_idx):
            self.P_l[:, idx] = load[:, i % load.shape[1]]
        self.Q_l = 0.5 * self.P_l
        # Convert loads to p.u.
        self.P_l /= Sbase;
        self.Q_l /= Sbase;

        # Generation
        solar_rad = scipy.io.loadmat(f'{dataPath}/Irradiance_1sec.mat')
        solar_rad = solar_rad['Irr24_seq'].transpose() # # 604800 x 1

        self.gen_idx = np.array([4, 7, 9, 10, 11, 13, 16, 17, 20, 22, 23, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36])-1

        #% PV capacity [kVA]
        self.max_S = np.array([200, 200, 100, 200, 200, 200, 200, 200, 200, 200, 200,
                    200, 200, 200, 200, 200, 200, 350, 350, 300, 300]);
        self.max_S = self.max_S * 1000 / Sbase # Convert to p.u.
        #% Area of the PV array
        Area_PV = np.array([100, 100, 100, 200, 200, 200, 200, 200, 200, 100,
           200, 200, 200, 100, 200, 200, 200, 350, 350, 300, 300]);
        #% PV efficiency;
        PV_Irradiance_to_Power_Efficiency = 1;

        self.P_gen = np.zeros((load.shape[0], self.n))
        gen = solar_rad * Area_PV * PV_Irradiance_to_Power_Efficiency
        gen /= Sbase # Convert to p.u.
        self.P_gen[:, self.gen_idx] = gen.clip(max = self.max_S.reshape(1, -1))
    
