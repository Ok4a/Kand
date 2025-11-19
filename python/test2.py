import matplotlib.pyplot as plt
import numpy as np

r_lap = np.loadtxt("r_norm_lap_r.txt")
s_lap = np.loadtxt("r_norm_lap_s.txt")
r_desk = np.loadtxt("r_norm_desk_r.txt")
s_desk = np.loadtxt("r_norm_desk_s.txt")

plt.plot((range(len(r_lap))), np.log(r_lap), label = "r_lap")
plt.plot((range(len(s_lap))), np.log(s_lap), label = "s_lap")
plt.plot((range(len(r_desk))), np.log(r_desk), label = "r_desk")
plt.plot((range(len(s_desk))), np.log(s_desk), label = "s_desk")
plt.legend()
plt.title("Log Residual norm")
plt.ylabel("log residual norm")
plt.xlabel("Iteration Count")
plt.show()