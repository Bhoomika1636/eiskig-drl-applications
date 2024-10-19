from eta_utility.simulators import fmu
import matplotlib.pyplot as plt
import numpy as np
print("TestFMU")
config = {"_id": 1,
          "fmu_path": "C:\Gitlab\experimentshr_ast\experiments_hr\Askari_RL_Test1\environment\RL_Test_WT.fmu",
          "stop_time": 1000
          }
simulator = fmu.FMUSimulator(_id=1, fmu_path="C:\Gitlab\experimentshr_ast\experiments_hr\Askari_RL_Test1\environment\RL_Test_WT.fmu", stop_time=100, names_inputs=["u_VR"], names_outputs=["T_in_cooling"],
                             init_values={"u_VR":0.0})
# simulator2 = fmu.FMUSimulator(config)
simulator.set_values({"u_VR": 0.0})
simulator.step_size = 1
output = []
for i in range(2500):
    if i == 30:
        simulator.set_values({"u_VR": 1.0})
    if i == 1500:
        simulator.set_values({"u_VR": 0.5})
    output.append(simulator.step())
out = np.array(output)
plt.figure(1)
plt.plot(np.arange(2500),out[:,0])
plt.figure(2)
plt.plot(np.arange(10,2500),out[10:2500,0])
plt.show()
print("end")
