"""One real follower arm on its Feetech bus, with the same interface as the simulator's arm: read and write the 5
joint angles (lerobot degrees) and the gripper opening (mm), torque on and off, servo temperatures.

Connecting only opens the port and pings the motors: nothing is written. Torque comes on in torque(True), which first
sets every goal to where the motor is now, so no servo jumps to an old goal.
"""
import json
import os

import numpy as np

from .kin import CALIB_DIR, JOINTS

GRIPPER_TORQUE = dict(Max_Torque_Limit=700, Protection_Current=350, Overload_Torque=40)  # lerobot's 500/250/25 slip
P_GAIN, D_GAIN = 16, 32
RETRIES = 2  # a garbled status packet is retried before it counts as a failed read


class CalibrationMismatch(Exception):
    pass


class BusArmIO:
    def __init__(self, port, calibration_id, stroke_mm, bus=None):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode

        self.stroke = float(stroke_mm)
        self.names = JOINTS + ["gripper"]
        if bus is None:
            from lerobot.motors.feetech import FeetechMotorsBus

            cal = json.load(open(os.path.join(CALIB_DIR, f"{calibration_id}.json")))
            motors = {j: Motor(cal[j]["id"], "sts3215", MotorNormMode.DEGREES) for j in JOINTS}
            motors["gripper"] = Motor(cal["gripper"]["id"], "sts3215", MotorNormMode.RANGE_0_100)
            bus = FeetechMotorsBus(port=port, motors=motors,
                                   calibration={k: MotorCalibration(**v) for k, v in cal.items()})
            bus.connect()
            if not bus.is_calibrated:
                bus.disconnect(disable_torque=False)
                raise CalibrationMismatch(f"the motors on {port} don't hold the calibration in {calibration_id}.json: "
                                          "run lerobot-calibrate (and the gripper limits) first")
        self.bus = bus

    def read(self):
        pos = self.bus.sync_read("Present_Position", num_retry=RETRIES)
        return np.array([pos[j] for j in JOINTS] + [pos["gripper"] / 100 * self.stroke], dtype=float)

    def write(self, goal):
        g = {j: float(goal[i]) for i, j in enumerate(JOINTS)}
        g["gripper"] = float(np.clip(goal[5] / self.stroke * 100, 0, 100))
        self.bus.sync_write("Goal_Position", g, num_retry=RETRIES)

    def torque(self, on):
        if not on:
            self.bus.disable_torque()
            return
        from lerobot.motors.feetech import OperatingMode

        self.bus.disable_torque()
        self.bus.configure_motors()
        for m in self.names:
            self.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", m, P_GAIN)
            self.bus.write("I_Coefficient", m, 0)
            self.bus.write("D_Coefficient", m, D_GAIN)
        for reg, v in GRIPPER_TORQUE.items():
            self.bus.write(reg, "gripper", v)
        here = self.bus.sync_read("Present_Position")
        self.bus.sync_write("Goal_Position", here)
        self.bus.enable_torque()

    def temperatures(self):
        t = self.bus.sync_read("Present_Temperature", normalize=False, num_retry=RETRIES)
        return [float(t[m]) for m in self.names]

    def close(self, torque_off=True):
        self.bus.disconnect(disable_torque=torque_off)
