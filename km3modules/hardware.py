# Filename: hardware.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
A collection of controllers and hardware related stuff.

"""

import time
import os

import km3pipe as kp

__author__ = "Jonas Reubelt and Tamas Gal"
__email__ = "jreubelt@km3net.de"
__status__ = "Development"

log = kp.logger.get_logger(__name__)  # pylint: disable=C0103


class PhidgetsController(kp.Module):
    def configure(self):
        from Phidgets.Devices.Stepper import Stepper
        from Phidgets.Devices.Encoder import Encoder

        self.current_limit = self.get("current_limit") or 2.5
        self.motor_id = self.get("motor_id") or 0
        self.stepper = Stepper()
        self.encoder = Encoder()
        self.setup(self.motor_id)

    def setup(self, motor_id):
        self.stepper.openPhidget()
        self.stepper.waitForAttach(10000)

        self.encoder.openPhidget()
        self.encoder.waitForAttach(10000)

        self.stepper.setVelocityLimit(motor_id, 1000)
        self.stepper.setAcceleration(motor_id, 5000)
        self.stepper.setCurrentLimit(motor_id, 2.5)

        self.e = 13250.0
        self.s = 70500.0

        self._stepper_dest = 0
        self._encoder_dest = 0

        self.reset_positions()

    def drive_to_angle(self, ang, motor_id=0, relative=False):
        stepper_dest = self._stepper_dest = self.raw_stepper_position(ang)
        self._encoder_dest = self.raw_encoder_position(ang)

        if relative:
            self.reset_positions()

        self.wake_up()
        time.sleep(0.1)

        self.stepper_target_pos = stepper_dest
        self.wait_for_stepper()
        self.log_offset()

        while abs(self.offset) > 1:
            self.log_offset()
            stepper_offset = round(self.offset / self.e * self.s)
            log.debug("Correcting stepper by {0}".format(stepper_offset))
            log.debug("Stepper target pos: {0}".format(self.stepper_target_pos))
            log.debug("Stepper pos: {0}".format(self.stepper_pos))
            self.stepper_target_pos = self.stepper_pos + stepper_offset
            self.wait_for_stepper()

        self.log_positions()
        self.stand_by()

    def wait_for_stepper(self):
        while self.stepper_pos != self._stepper_dest:
            time.sleep(0.1)
            self.log_positions()

    def log_offset(self):
        log.debug("Difference (encoder): {0}".format(self.offset))

    @property
    def offset(self):
        return self._encoder_dest - self.encoder_pos

    @property
    def stepper_target_pos(self):
        return self.stepper.getTargetPosition(self.motor_id)

    @stepper_target_pos.setter
    def stepper_target_pos(self, val):
        self._stepper_dest = int(val)
        self.stepper.setTargetPosition(self.motor_id, int(val))

    @property
    def stepper_pos(self):
        return self.stepper.getCurrentPosition(self.motor_id)

    @stepper_pos.setter
    def stepper_pos(self, val):
        self.stepper.setCurrentPosition(self.motor_id, int(val))

    @property
    def encoder_pos(self):
        return self.encoder.getPosition(self.motor_id)

    def raw_stepper_position(self, angle):
        return round(angle * self.s / 360)

    def raw_encoder_position(self, angle):
        return round(angle * self.e / 360)

    def wake_up(self, motor_id=0):
        self.stepper.setEngaged(motor_id, 1)

    def stand_by(self, motor_id=0):
        self.stepper.setEngaged(motor_id, 0)

    def reset_positions(self, motor_id=0):
        self.stepper.setCurrentPosition(motor_id, 0)
        self.encoder.setPosition(motor_id, 0)

    def log_positions(self, motor_id=0):
        log.info(
            "Stepper position: {0}\nEncoder position:{1}".format(
                self.stepper_pos / self.s * 360, self.encoder_pos / self.e * 360
            )
        )


class USBTMC(object):
    "USB TMC communicator"

    def __init__(self, path):
        self.device = os.open(path, os.O_RDWR)

    def write(self, msg):
        os.write(self.device, msg)

    def read(self, size=1000):
        return os.read(self.device, size)

    @property
    def name(self):
        self.write(b"*IDN?")
        return self.read()

    def reset(self):
        self.write(b"*RST")


class Agilent33220A(object):
    """Controller for the Arbitrary Waveform Generator"""

    def __init__(self, path):
        self.tmc = USBTMC(path)
        self._output = False
        self._amplitude = None
        self._frequency = None
        self._mode = None

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self.tmc.write("OUTP {0}".format("ON" if value else "OFF").encode())
        self._output = value

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, val):
        low, high = val
        diff = high - low
        offset = diff / 2
        self.tmc.write("VOLT:OFFS {0}".format(offset).encode())
        self.tmc.write("VOLT {0}".format(diff).encode())
        self._amplitude = val
        self._mode = None

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, val):
        self.tmc.write("FREQ {0}".format(val).encode())
        self._frequency = val

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        valid_modes = ("sin", "squ", "ramp", "puls", "nois", "dc", "user")
        if val not in valid_modes:
            print(
                "Not a valid mode: '{0}'. Valid modes are: {1}".format(val, valid_modes)
            )
            return
        self.tmc.write("FUNC {0}".format(val.upper()).encode())
        self._mode = val
