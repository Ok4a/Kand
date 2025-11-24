import matplotlib.pyplot as plt
import numpy as np

r_lap = np.loadtxt("r_norm_lap_r.txt")
s_lap = np.loadtxt("r_norm_lap_s.txt")

r_lap_mort = np.loadtxt("r_norm_lap_r_mort.txt")
s_lap_mort = np.loadtxt("r_norm_lap_s_mort.txt")

print(np.linalg.norm(r_lap-r_lap_mort))


r_lap_u = np.loadtxt("r_norm_lap_r_u.txt")
s_lap_u = np.loadtxt("r_norm_lap_s_u.txt")
# r_desk = np.loadtxt("r_norm_desk_r.txt")
# s_desk = np.loadtxt("r_norm_desk_s.txt")

plt.plot((range(len(r_lap))), np.log(r_lap), label = "r_lap")
plt.plot((range(len(s_lap))), np.log(s_lap), label = "s_lap")
plt.plot((range(len(r_lap_mort))), np.log(r_lap_mort), label = "r_mort")
plt.plot((range(len(s_lap_mort))), np.log(s_lap_mort), label = "s_mort")
plt.plot((range(len(r_lap_u))), np.log(r_lap_u), label = "r_lap_u")
plt.plot((range(len(s_lap_u))), np.log(s_lap_u), label = "s_lap_u")
plt.legend()
plt.title("Log Residual norm")
plt.ylabel("log residual norm")
plt.xlabel("Iteration Count")
plt.show()