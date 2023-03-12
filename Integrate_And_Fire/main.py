import AlphaSynapseIF
import IntegrateAndFire
import numpy as np
import matplotlib.pyplot as plt

def simulate_asif():
    model = AlphaSynapseIF.AlphaSynapseIF()
    
    # simulation
    iteration = 200
    spike_train = np.random.rand(iteration) > 0.9

    res = []
    for t in range(iteration):
      model.update(t, spike_train)
      res += [model.V]

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(np.arange(0,iteration), spike_train)
    axs[0].set_title('Input spike train')
    axs[1].plot( np.arange(0,iteration) , res )
    plt.draw()
    axs[1].set_title('Output spike train')
    fig.savefig('example.png')

def simulate_if():
    model = IntegrateAndFire.IntegrateAndFire(1, 40, 0, 10, 50)
    inputCurrent = 1000 / 1000 # nA

    print("Simulation Started")
    res = []  # voltage trace for plotting
    iteration = 100

    for t in range(iteration):
      model.update(inputCurrent)
      if(model.fired):
        print(f"{t} fired")
      res += [model.membranePotential]
    print("Simulation Over")

    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111)
    ax.plot(res)
    fig.savefig('example.png')

if __name__ == '__main__':
    simulate_if()
    # simulate asif()
