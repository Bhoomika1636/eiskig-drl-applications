from eta_utility.simulators import fmu
import matplotlib.pyplot as plt
import numpy as np
print("TestFMU")
config = {"_id": 1,
          "fmu_path": "C:\Gitlab\experimentshr_ast\experiments_hr\Askari_MA\FMU_TEST_KTS.fmu",
          "stop_time": 1000
          }
simulator = fmu.FMUSimulator(_id=1, fmu_path="C:\Gitlab\experimentshr_ast\experiments_hr\Askari_MA\FMU_TEST_KTS.fmu", stop_time=100, names_inputs=["u_KT"], names_outputs=["P_el","T_out"])
# simulator2 = fmu.FMUSimulator(config)
simulator.set_values({"u_KT": 0.0})
simulator.step_size = 1
output = []
for i in range(2500):
    if i == 30:
        simulator.set_values({"u_KT": 1.0})
    if i == 1500:
        simulator.set_values({"u_KT": 0.5})
    output.append(simulator.step())
out = np.array(output)
plt.figure(1)
plt.plot(np.arange(2500),out[:,1])
plt.figure(2)
plt.plot(np.arange(10,2500),out[10:2500,0])
plt.show()
print("end")
