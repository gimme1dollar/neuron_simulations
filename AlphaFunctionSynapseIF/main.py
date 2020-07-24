import AlphaSynapseIF
import numpy as np
import matplotlib.pyplot as plt

# Model Initiation
model = AlphaSynapseIF.AlphaSynapseIF()


# Simulation
iteration = 200
spike_train = np.random.rand(iteration) > 0.9

res = []
for t in range(iteration):
  model.update(t, spike_train)
  res += [model.V]

# Plot Graph
fig, axs = plt.subplots(2, 1)
#fig = plt.figure(figsize = (10,8))
#axs = fig.add_subplot(111)
axs[0].plot(np.arange(0,iteration), spike_train)
axs[0].set_title('Input spike train')
#ax.plot(V_trace)
axs[1].plot( np.arange(0,iteration) , res )
plt.draw()
axs[1].set_title('Output spike train')
fig.savefig('example.png')
