import numpy as np

class IntegrateAndFire:
  # Basic integrate-and-fire neuron {R Rao, 2007}
  def __init__(self, c, r, v_0, v_th, v_sp):
    self.capacitance = c
    self.resistance = r

    self.membranePotential = v_0
    self.thresholdVoltage = v_th
    self.spikeVoltage = v_sp
    self.resetVoltage = v_th * 0.2

    self.fired = 0
  def update(self, i):
    if(not self.fired) :
      # dV/dt = - V/RC + I/C
      self.membranePotential = self.membranePotential - (self.membranePotential/(self.capacitance * self.resistance)) + i/self.capacitance

      self.fired = 0
    else :
      self.fired -= 1
      self.membranePotential = self.resetVoltage # reset voltage
    
    if(self.membranePotential > self.thresholdVoltage) :
        self.membranePotential = self.spikeVoltage
        self.fired = 5
