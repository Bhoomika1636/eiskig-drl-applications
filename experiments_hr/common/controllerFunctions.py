import pandas as pd

class Hysteresis:
    def __init__(self, th_low, th_high,begin=True):
        """
        Initializes a Hysteresis instance with upper and lower thresholds.

        Parameters:
        - th_low: Lower threshold for the hysteresis.
        - th_high: Upper threshold for the hysteresis.
        """
        self.th_low = th_low
        self.th_high = th_high
        self.state = begin

    def update(self, observation):
        """
        Updates the hysteresis state based on the given observation.

        Parameters:
        - observation: The input observation value.

        Notes:
        - If the observation is greater than or equal to the upper threshold, the state is set to True.
        - If the observation is less than or equal to the lower threshold, the state is set to False.
        - If the observation is between the lower and upper thresholds, the current state is maintained.
        """
        if observation >= self.th_high:
            self.state = True
        elif observation <= self.th_low:
            self.state = False
        # If observation is between th_low and th_high, maintain the current state

    def get_state(self):
        """
        Returns the current state of the hysteresis.

        Returns:
        - state: Boolean value representing the current state of the hysteresis.
        """
        return self.state


class DualHysteresis:
    def __init__(self, th_low, th_high, tl_low, tl_high):
        """
        Initializes a DualHysteresis instance with upper and lower thresholds for two hysteresis instances.

        Parameters:
        - th_low: Lower threshold for the upper hysteresis.
        - th_high: Upper threshold for the upper hysteresis.
        - tl_low: Lower threshold for the lower hysteresis.
        - tl_high: Upper threshold for the lower hysteresis.
        """
        self.upper_hysteresis = Hysteresis(th_low, th_high)
        self.lower_hysteresis = Hysteresis(tl_low, tl_high)

    def update(self, observation):
        """
        Updates both hysteresis instances based on the given observation and returns the AND connection of their states.

        Parameters:
        - observation: The input observation value.

        Returns:
        - output: Boolean value representing the AND connection of upper and lower hysteresis states.
        """
        self.upper_hysteresis.update(observation)
        self.lower_hysteresis.update(observation)

        # Calculate and return the AND connection of both hysteresis states
        return self.upper_hysteresis.get_state() and self.lower_hysteresis.get_state()


class ControlKT:
    def __init__(self, sampling_time) -> None:
        self.sampling_time = sampling_time
        self.tank_level_ok = Hysteresis(0.1, 0.3)
        data = {
            "Stufe": [
                "gw_pump_führung",
                "gw_KT_führung_stufe1",
                "gw_KT_führung_stufe2",
                "gw_pump_folge1",
                "gw_KT_folge1_stufe1",
                "gw_KT_folge1_stufe2",
                "gw_pump_folge2",
                "gw_KT_folge2_stufe1",
                "gw_KT_folge2_stufe2",
            ],
            "th_high": [15, 30, 40, 50, 60, 70, 80, 90, 95],
            'time_thres': [120, 30, 120,120,120,120,120,120,120]
        }
        self.df = pd.DataFrame(data)
        self.df['isOn'] = False
        self.df['levelSafeIsOn'] = True
        self.df['th_low'] = self.df['th_high'] - 13
        self.df['timer'] = 0

        self.T_cw_tank = 0
        self.level_ww_tank = 0

        self.Yh = 0
        self.prevYh = 0

        self.XS_steilheit_regler = 20
        self.T_soll = 14
        pass

    def set_T_soll(self, T_soll):
        self.T_soll = T_soll

    def update(self, T_cw_tank, level_ww_tank):
        self.prevYh = self.Yh

        self.T_cw_tank = T_cw_tank
        self.level_ww_tank = level_ww_tank
        self.Yh = 50 + self.XS_steilheit_regler * (self.T_cw_tank - self.T_soll)

        for index, row in self.df.iterrows():
            stufe = row['Stufe']
            th_low = row['th_low']
            th_high = row['th_high']
            time_thres = row['time_thres']
            isOn = row['isOn']
            
            if (not isOn) and self.Yh > th_high:
                self.df.at[index, 'timer'] += self.sampling_time  # Increment timer
            elif isOn and self.Yh < th_low:
                self.df.at[index, 'timer'] += self.sampling_time  # Increment timer
            else:
                # Y_h not over a threshold range
                self.df.at[index, 'timer'] = 0  # Reset timer

            if self.df.at[index, 'timer'] > time_thres:
                # Timer exceeds time_thres, set isOn to True
                self.df.at[index, 'isOn'] = not self.df.at[index, 'isOn']
                self.df.at[index, 'timer'] = 0  # Reset timer
            
        if self.tank_level_ok.update(level_ww_tank): # TODO: what if level > max=1.8 and fix current hysteresis !
            self.df['levelSafeIsOn'] = False
        else:
            self.df['levelSafeIsOn'] = self.df['isOn']

    def getControlls(self):
        return self.df.set_index('Stufe')['levelSafeIsOn'].to_dict()

class ControlKKM:
    def __init__(self, sampling_time) -> None:
        self.sampling_time = sampling_time
        self.T_soll = 8
        self.T_min = 6
        self.T_max = 12
        self.timer = 0
        self.KKM1_on = True
        self.KKM2_on = False
        self.time_thres = 30*60  # 30 minutes
        pass

    def update(self, T_in_cold, T_out_cold):
        T_spreizung = abs(T_in_cold - T_out_cold)
        if T_spreizung>=3 and not self.KKM2_on:
            self.timer += self.sampling_time
        elif self.KKM2_on and T_spreizung<3:
            self.timer += self.sampling_time
        else:
            self.timer = 0
            
        if self.timer > self.time_thres:
            self.KKM2_on = not self.KKM2_on
            self.timer = 0
        kkm1on = 1 if self.KKM1_on else 0
        kkm2on = 1 if self.KKM2_on else 0
        return kkm1on, kkm2on