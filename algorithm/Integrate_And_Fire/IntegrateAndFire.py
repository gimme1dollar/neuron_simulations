import numpy as np

class IntegrateAndFire:
  # Basic integrate-and-fire neuron {R Rao, 2007}
  def __init__(self, args, **kwargs):
    self.capacitance = kwargs['c']
    self.resistance = kwargs['r']

    self.membranePotential = kwargs['v_0']
    self.thresholdVoltage = kwargs['v_th']
    self.spikeVoltage = kwargs['v_sp']
    self.resetVoltage = kwargs['v_th'] * 0.2

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

def main(args, **kwargs):
    model = IntegrateAndFire(args, **kwargs)

    epoch = args.epoch
    for i in range(epoch):
        model.update(i)
        print(model.membranePotential)
