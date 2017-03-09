# coding=utf-8
# Filename: hardware.py
# pylint: disable=locally-disabled
"""
A collection of controllers and hardware related stuff.

"""
from __future__ import division, absolute_import, print_function

import time

import os

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

    def drive_angle(self, ang, motor_id=0):
        stepper_dest = int(ang * self.s / 360.)
        encoder_dest = abs(int(stepper_dest / self.s * self.e))
        self.wake_up()
        self.reset_positions()
        time.sleep(0.1)
        self.stepper.setTargetPosition(motor_id, stepper_dest)
        while abs(self.encoder.getPosition(motor_id)) < encoder_dest:
            while abs(self.stepper.getCurrentPosition(motor_id)) < abs(stepper_dest):
                time.sleep(0.1)
                self.log_positions()
            self.stepper.setCurrentPosition(motor_id, int(self.encoder.getPosition(0) / self.e * self.s))
            self.stepper.setTargetPosition(motor_id, stepper_dest)
            time.sleep(0.1)

        self.log_positions()
        self.stand_by()

    def wake_up(self, motor_id=0):
        self.stepper.setEngaged(motor_id, 1)

    def stand_by(self, motor_id=0):
        self.stepper.setEngaged(motor_id, 0)

    def reset_positions(self, motor_id=0):
        self.stepper.setCurrentPosition(motor_id, 0)
        self.encoder.setPosition(motor_id, 0)

    def log_positions(self, motor_id=0):
        log.info("Stepper position: {0}\nEncoder position:{1}"
                 .format(self.stepper.getCurrentPosition(motor_id) / self.s * 360,
                         self.encoder.getPosition(motor_id) / self.e * 360))
