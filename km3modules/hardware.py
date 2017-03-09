# coding=utf-8
# Filename: hardware.py
# pylint: disable=locally-disabled
"""
A collection of controllers and hardware related stuff.

"""
from __future__ import division, absolute_import, print_function

import time

import km3pipe as kp

__author__ = "Jonas Reubelt and Tamas Gal"
__email__ = "jreubelt@km3net.de"
__status__ = "Development"

log = kp.logger.logging.getLogger(__name__)  # pylint: disable=C0103


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

        self.e = 13250.
        self.s = 70500.

        self.reset_positions()

    def drive_angle(self, ang, motor_id=0, relative=False):
        stepper_dest = self.raw_stepper_position(ang)
        encoder_dest = self.raw_encoder_position(ang)
        if relative:
            self.reset_positions()
        self.wake_up()
        time.sleep(0.1)

        self.stepper_target_pos = stepper_dest

        while abs(self.encoder_pos - encoder_dest) > 5:
            log.info("Difference: ".format(self.encoder_pos - encoder_dest))
            while self.stepper_pos != stepper_dest:
                time.sleep(0.1)
                self.log_positions()
            self.stepper_pos = stepper_dest

        self.log_positions()
        self.stand_by()

    @property
    def stepper_target_pos(self):
        return self.stepper.getTargetPosition(self.motor_id)

    @stepper_target_pos.setter
    def stepper_target_pos(self, val):
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
        log.info("Stepper position: {0}\nEncoder position:{1}"
                 .format(self.stepper_pos / self.s * 360,
                         self.encoder_pos / self.e * 360))
