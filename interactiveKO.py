%matplotlib auto
import matplotlib.pyplot as plt
fig, ax = plt.subplots() # Diagram will pop up. Let’s interact.
ln, = ax.plot(range(5))  # Drawing a line
ln.set_color("orange")   # Changing drawn line to orange
plt.ioff() # Stopped interaction
ln.set_color("red")
plt.ion() # Started interaction
