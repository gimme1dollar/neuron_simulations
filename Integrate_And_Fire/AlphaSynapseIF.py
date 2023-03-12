import time
import numpy as np
from numpy import concatenate as cc

np.random.seed(0)

class AlphaSynapseIF:
  ''' 
  (C) {R Rao 2007}
  '''

  def __init__(self, functionDuration = 100, E_leak = -60):
    # Alpha Function Synaptic Conductance
    t_step = 1
    t_peak = 5 # ms
    g_peak = 0.05 # nS (peak synaptic conductance)
    const = g_peak / (t_peak*np.exp(-1));
    
    self.synConductanceMaxDuration = functionDuration # Max duration of syn conductance
    t_vec = np.arange(0, self.synConductanceMaxDuration + t_step, t_step)
    self.alpha_func = const * t_vec * (np.exp(-t_vec/t_peak))

    # Model Parameter
    self.C = 0.5 # nF
    self.R = 40 # M ohms

    self.g_ad = 0
    self.G_inc = 1
    self.tau_ad = 2

    self.E_leak = E_leak # mV, equilibrium potential
    self.E_syn = 0 # Excitatory synapse (why is this excitatory?)
    self.g_syn = 0 # Current syn conductance

    self.V_th = -40 # spike threshold mV
    self.V_spike = 50 # spike value mV

    self.t_list = np.array([], dtype=int)
    self.V = E_leak

    self.fired = 0 # Starting value of ref period counter

  def update(self, t, inputTrain):
    if inputTrain[t]: # check for input spike
        self.t_list = cc([self.t_list, [1]])

    # Calculate synaptic current due to current and past input spikes
    self.g_syn = np.sum(self.alpha_func[self.t_list])
    self.I_syn = self.g_syn*(self.E_syn - self.V) 

    # Update spike times
    if np.any(self.t_list):
        self.t_list = self.t_list + 1
        if self.t_list[0] == self.synConductanceMaxDuration: # Reached max duration of syn conductance
            self.t_list = self.t_list[1:]

    # Compute membrane voltage
    # Euler method: V(t+h) = V(t) + h*dV/dt
    if not self.fired:
        self.V = self.V + (-((self.V - self.E_leak)*(1 + self.R * self.g_ad)/(self.R * self.C)) + (self.I_syn/self.C))
        self.g_ad = self.g_ad + (-self.g_ad/self.tau_ad) # spike rate adaptation
    else:
        self.fired -= 1
        self.V = self.V_th - 10 # reset voltage after spike
        self.g_ad = 0

    # Generate spike
    if (self.V > self.V_th) and not self.fired:
        self.V = self.V_spike
        self.fired = 4
        self.g_ad = self.g_ad + self.G_inc