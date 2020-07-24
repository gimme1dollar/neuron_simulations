import IntegrateAndFire
import matplotlib.pyplot as plt

# Model Initiation
model = IntegrateAndFire.IntegrateAndFire(1, 40, 0, 10, 50)
inputCurrent = 1000 / 1000 # nA


# Simulation
print("Simulation Started")
res = []  # voltage trace for plotting
iteration = 100

for t in range(iteration):
  model.update(inputCurrent)
  if(model.fired):
    print(f"{t} fired")
  res += [model.membranePotential]
print("Simulation Over")


# Plot
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax.plot(res)
fig.savefig('example.png')
