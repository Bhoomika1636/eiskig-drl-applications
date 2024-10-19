from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from eta_utility.eta_x.agents import RuleBased

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv

# import sys
# sys.path.append("experiments_hr")


class HysteresisController:
    def __init__(self, hysteresis_range, target, inverted=False, init_value=0):
        self.hysteresis_range = hysteresis_range  # This is the hysteresis range
        self.target = target  # This is the target temperature
        self.inverted = inverted  # This should be True e.g. for eChiller, which should be 1 when T is too high
        self.output = init_value  # The output is always init with 0

    def update(self, actual_value):
        if self.inverted == False:
            if self.output == 0:  # controller is off
                # controller output still 0 if input value below threshhold
                self.output = 1 if actual_value <= self.target - self.hysteresis_range / 2 else 0
            else:  # controller is on
                self.output = 0 if actual_value >= self.target + self.hysteresis_range / 2 else 1

        else:
            if self.output == 0:  # controller is off
                # controller output still 0 if input value below threshhold
                self.output = 1 if actual_value >= self.target + self.hysteresis_range / 2 else 0
            else:  # controller is on
                self.output = 0 if actual_value <= self.target - self.hysteresis_range / 2 else 1

        return self.output

    def change_hysteresis_range(self, hysteresis_range):
        self.hysteresis_range = hysteresis_range

    def change_target(self, target):
        self.target = target


class RuleBasedController(RuleBased):
    """
    Simple rule based controller for supplysystem_a.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
    """

    def __init__(self, policy: type[BasePolicy], env: VecEnv, verbose: int = 1, **kwargs: Any):
        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        # extract action and observation names from the environments state_config
        self.action_names = self.env.envs[0].state_config.actions
        self.observation_names = self.env.envs[0].state_config.observations

        # set initial state

        self.initial_state = np.zeros(self.action_space.shape)

        self.init_Hysterese_Controllers()

    def init_Hysterese_Controllers(self):
        fTargetTemperature_HNHT = 70
        fTargetTemperature_HNLT_Cooling = 40
        fTargetTemperature_HNLT_Heating = 35
        fTargetTemperature_CN = 18

        self.Controller_CHP_Prio = HysteresisController(hysteresis_range=4, target=15)
        self.Controller_CHP1 = HysteresisController(hysteresis_range=0, target=0)  # this is set later
        self.Controller_CHP2 = HysteresisController(hysteresis_range=0, target=0)  # this is set later
        self.Controller_CondensingBoiler = HysteresisController(hysteresis_range=10, target=fTargetTemperature_HNHT + 1)

        self.Controller_VSI_Unloading = HysteresisController(hysteresis_range=4, target=fTargetTemperature_HNHT - 0)

        self.Controller_VSI_Unloading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNHT, inverted=True
        )
        self.Controller_VSI_Loading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNHT
        )

        self.Controller_OuterCapillaryTubeMats = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling + 2, inverted=True  # 42
        )  # cooling application
        self.Controller_OuterCapillaryTubeMats_Permission = HysteresisController(
            hysteresis_range=6, target=0
        )  # target is set later

        self.Controller_HeatExchanger1 = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating + 4
        )  # 39

        self.Controller_eChiller = HysteresisController(hysteresis_range=6, target=fTargetTemperature_CN, inverted=True)

        self.Controller_HeatPump = HysteresisController(hysteresis_range=5, target=fTargetTemperature_HNLT_Heating + 0)
        self.Controller_HeatPump_Permission = HysteresisController(
            hysteresis_range=3, target=fTargetTemperature_CN - 3, inverted=True
        )

        self.Controller_HVFA_CN_Loading_Permission = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN, inverted=True
        )
        self.Controller_HVFA_CN_Unloading_Permission = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN
        )
        self.Controller_Buffer_HVFA_CN_Loading = HysteresisController(hysteresis_range=2, target=fTargetTemperature_CN)
        self.Controller_Buffer_HVFA_CN_Unloading = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN, inverted=True
        )

        self.Controller_Controller_HVFA_HNLT_Loading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling
        )
        self.Controller_Controller_HVFA_HNLT_Unloading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating, inverted=True
        )
        self.Controller_Buffer_HVFA_HNLT_Loading = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling, inverted=True
        )
        self.Controller_Buffer_HVFA_HNLT_Unloading = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating + 2
        )

    # def control_rules(self, observation: np.ndarray) -> np.ndarray:
    def control_rules(self, observation) -> np.ndarray:
        """
        Controller of the model.

        :param observation: Observation from the environment.
        :returns: actions
        """

        # print("step performed")
        # print(type(observation))
        observation = dict(zip(self.observation_names, observation))

        action = dict.fromkeys(self.action_names, 0)  # this creates a dict with the action names and 0 as values

        # get observations
        HNHT_Buffer_fMidTemperature = observation["HNHT_Buffer_fMidTemperature"]
        HNLT_Buffer_fMidTemperature = observation["HNLT_Buffer_fMidTemperature"]
        CN_Buffer_fMidTemperature = observation["CN_Buffer_fMidTemperature"]

        VSI_fUpperTemperature = observation["HNHT_VSI_fUpperTemperature"]
        VSI_fLowerTemperature = observation["HNHT_VSI_fLowerTemperature"]

        # CN_HVFA_fUpperTemperature = observation["CN_HVFA_fUpperTemperature"]
        CN_HVFA_fLowerTemperature = observation["CN_HVFA_fLowerTemperature"]
        HNLT_HVFA_fLowerTemperature = observation["HNLT_HVFA_fLowerTemperature"]

        T_Mean = observation["weather_T_amb_Mean"]
        T_amb = observation["weather_T_amb"]

        # define parameters
        fTargetTemperature_HNHT = 70

        # ask for controller outputs for actions which are more complex
        if self.Controller_CHP_Prio.update(actual_value=T_Mean) == 1:
            CHP1_hysteresis_range = 14
            CHP2_hysteresis_range = 12
            fOffset_TargetTemperature_CHP1 = 1
            fOffset_TargetTemperature_CHP2 = 0

            # change hysteresis settings
            self.Controller_CHP1.change_hysteresis_range(CHP1_hysteresis_range)
            self.Controller_CHP1.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP1)

            self.Controller_CHP2.change_hysteresis_range(CHP2_hysteresis_range)
            self.Controller_CHP2.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP2)
        else:
            CHP1_hysteresis_range = 12
            CHP2_hysteresis_range = 14
            fOffset_TargetTemperature_CHP1 = 0
            fOffset_TargetTemperature_CHP2 = 1
            self.Controller_CHP1.change_hysteresis_range(CHP1_hysteresis_range)
            self.Controller_CHP1.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP1)

            self.Controller_CHP2.change_hysteresis_range(CHP2_hysteresis_range)
            self.Controller_CHP2.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP2)

        # set actions
        action["bSetStatusOn_CHP1"] = self.Controller_CHP1.update(actual_value=HNHT_Buffer_fMidTemperature)
        action["bSetStatusOn_CHP2"] = self.Controller_CHP2.update(actual_value=HNHT_Buffer_fMidTemperature)

        action["bSetStatusOn_CondensingBoiler"] = self.Controller_CondensingBoiler.update(
            actual_value=HNHT_Buffer_fMidTemperature
        )

        # print("HNHT_Mid_T:", HNHT_Buffer_fMidTemperature)
        # print("Action Boiler:", action["bSetStatusOn_CondensingBoiler"])

        # VSI
        Controller_Output_VSI_Unloading = self.Controller_VSI_Unloading.update(actual_value=HNHT_Buffer_fMidTemperature)
        Controller_Output_VSI_Unloading_Permission = self.Controller_VSI_Unloading_Permission.update(
            actual_value=VSI_fUpperTemperature
        )
        Controller_Output_VSI_Loading_Permission = self.Controller_VSI_Loading_Permission.update(
            actual_value=VSI_fLowerTemperature
        )

        if Controller_Output_VSI_Unloading and Controller_Output_VSI_Unloading_Permission:
            action["bSetStatusOn_VSIStorage"] = 1
        elif action["bSetStatusOn_CHP1"] == 1 or action["bSetStatusOn_CHP2"] == 1:
            if Controller_Output_VSI_Unloading == 0 and Controller_Output_VSI_Loading_Permission == 1:
                action["bSetStatusOn_VSIStorage"] = 1
            else:
                action["bSetStatusOn_VSIStorage"] = 0
        else:
            action["bSetStatusOn_VSIStorage"] = 0

        if Controller_Output_VSI_Unloading == 1 and Controller_Output_VSI_Unloading_Permission == 1:
            action["bLoading_VSISystem"] = 0
        else:
            action["bLoading_VSISystem"] = 1

        # HNHT-HNLT Linkage
        action["bSetStatusOn_HeatExchanger1"] = self.Controller_HeatExchanger1.update(
            actual_value=HNLT_Buffer_fMidTemperature
        )
        # print("HNLT Temp", HNLT_Buffer_fMidTemperature)
        # print("PWT1:", action["bSetStatusOn_HeatExchanger1"])

        # HNLT
        Controller_Output_OuterCapillaryTubeMats = self.Controller_OuterCapillaryTubeMats.update(
            actual_value=HNLT_Buffer_fMidTemperature
        )
        self.Controller_OuterCapillaryTubeMats_Permission.change_target(HNLT_Buffer_fMidTemperature - 6)
        Controller_Output_OuterCapillaryTubeMats_Permission = self.Controller_OuterCapillaryTubeMats_Permission.update(
            actual_value=T_amb
        )
        if Controller_Output_OuterCapillaryTubeMats and Controller_Output_OuterCapillaryTubeMats_Permission:
            action["bSetStatusOn_OuterCapillaryTubeMats"] = 1
        else:
            action["bSetStatusOn_OuterCapillaryTubeMats"] = 0

        # HNLT HVFA
        Controller_Controller_HVFA_HNLT_Loading_Permission_Output = (
            self.Controller_Controller_HVFA_HNLT_Loading_Permission.update(HNLT_HVFA_fLowerTemperature)
        )
        Controller_Controller_HVFA_HNLT_Unloading_Permission_Output = (
            self.Controller_Controller_HVFA_HNLT_Unloading_Permission.update(HNLT_HVFA_fLowerTemperature)
        )
        Controller_Buffer_HVFA_HNLT_Loading_Output = self.Controller_Buffer_HVFA_HNLT_Loading.update(
            HNLT_Buffer_fMidTemperature
        )
        Controller_Buffer_HVFA_HNLT_Unloading_Output = self.Controller_Buffer_HVFA_HNLT_Unloading.update(
            HNLT_Buffer_fMidTemperature
        )

        if Controller_Controller_HVFA_HNLT_Loading_Permission_Output and Controller_Buffer_HVFA_HNLT_Loading_Output:
            action["bSetStatusOn_HVFASystem_HNLT"] = 1
        elif (
            Controller_Controller_HVFA_HNLT_Unloading_Permission_Output and Controller_Buffer_HVFA_HNLT_Unloading_Output
        ):
            action["bSetStatusOn_HVFASystem_HNLT"] = 1
        else:
            action["bSetStatusOn_HVFASystem_HNLT"] = 0

        if Controller_Controller_HVFA_HNLT_Loading_Permission_Output and Controller_Buffer_HVFA_HNLT_Loading_Output:
            action["bLoading_HVFASystem_HNLT"] = 1
        else:
            action["bLoading_HVFASystem_HNLT"] = 0

        # CN
        action["bSetStatusOn_eChiller"] = self.Controller_eChiller.update(CN_Buffer_fMidTemperature)

        # CN HVFA
        Controller_HVFA_CN_Loading_Permission_Output = self.Controller_HVFA_CN_Loading_Permission.update(
            CN_HVFA_fLowerTemperature
        )
        Controller_HVFA_CN_Unloading_Permission_Output = self.Controller_HVFA_CN_Unloading_Permission.update(
            CN_HVFA_fLowerTemperature
        )

        Controller_Buffer_HVFA_CN_HVFA_Loading_Output = self.Controller_Buffer_HVFA_CN_Loading.update(
            CN_Buffer_fMidTemperature
        )
        Controller_Buffer_HVFA_CN_HVFA_Unoading_Output = self.Controller_Buffer_HVFA_CN_Unloading.update(
            CN_Buffer_fMidTemperature
        )

        if Controller_HVFA_CN_Loading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Loading_Output:
            action["bSetStatusOn_HVFASystem_CN"] = 1
        elif Controller_HVFA_CN_Unloading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Unoading_Output:
            action["bSetStatusOn_HVFASystem_CN"] = 1
        else:
            action["bSetStatusOn_HVFASystem_CN"] = 0

        if Controller_HVFA_CN_Loading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Loading_Output:
            action["bLoading_HVFASystem_CN"] = 1
        else:
            action["bLoading_HVFASystem_CN"] = 0

        # HNLT-CN Linkage
        Controller_Output_HeatPump = self.Controller_HeatPump.update(HNLT_Buffer_fMidTemperature)
        Controller_Output_HeatPump_Permission = self.Controller_HeatPump_Permission.update(CN_Buffer_fMidTemperature)
        if Controller_Output_HeatPump and Controller_Output_HeatPump_Permission:
            action["bSetStatusOn_HeatPump"] = 1
        else:
            action["bSetStatusOn_HeatPump"] = 0

        # action["bSetStatusOn_HeatExchanger1"] = HeatExchanger1_action
        # # action["bSetStatusOn_CHP1"] = CHP1_action
        # # action["bSetStatusOn_CHP2"] = CHP2_action
        # # action["bSetStatusOn_CondensingBoiler"] = CondensingBoiler_action
        # # action["bSetStatusOn_VSIStorage"] = VSIStorage_action
        # # action["bLoading_VSISystem"] = bLoading_VSISystem_action
        # # action["bSetStatusOn_HVFASystem_HNLT"] = HVFASystem_HNLT_action
        # # action["bLoading_HVFASystem_HNLT"] = bLoading_HVFASystem_HNLT_action
        # # action["bSetStatusOn_eChiller"] = eChiller_action
        # # action["bSetStatusOn_HVFASystem_CN"] = HVFASystem_CN_action
        # # action["bLoading_HVFASystem_CN"] = bLoading_HVFASystem_CN_action
        # # action["bSetStatusOn_OuterCapillaryTubeMats"] = OuterCapillaryTubeMats_action
        # # action["bSetStatusOn_HeatPump"] = HeatPump_action
        # # action["HeatPump_Permission"] = Permission_action

        # print(action)
        actions = []
        actions.append(list(action.values()))
        # print(actions)
        actions = actions[0]

        return np.array(actions)
