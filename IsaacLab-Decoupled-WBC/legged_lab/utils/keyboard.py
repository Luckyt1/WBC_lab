# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""Keyboard controller for SE(2) control."""

import weakref
from collections.abc import Callable

import carb
import omni
import torch
from isaaclab.devices.device_base import DeviceBase

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.utils.student_observations import command_ranges, nominal_height


class Keyboard(DeviceBase):

    _TORSO_KEYS = {"H", "J", "Z", "X", "C", "V", "B", "N"}

    def __init__(self, env: BaseEnv, continuous_torso: bool = False, command_ranges_override=None):
        """Initialize the keyboard layer."""
        self.env = env
        ranges = command_ranges(command_ranges_override if command_ranges_override is not None else env)
        self._cmd_limits = {
            "cmd_x": ranges["lin_vel_x"],
            "cmd_y": ranges["lin_vel_y"],
            "cmd_z": ranges["ang_vel_z"],
            "hight": ranges["body_height"],
            "body_roll": ranges["body_roll"],
            "body_pitch": ranges["body_pitch"],
            "body_yaw": ranges["body_yaw"],
        }
        self._nominal_height = nominal_height(env)
        self._continuous_torso = continuous_torso
        self._held_torso_keys: set[str] = set()
        self._height_rate = 1.0  # m/s while held
        self._orientation_rate = 4.0  # rad/s while held
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        self.cmd_x = 0.0
        self.cmd_y = 0.0
        self.cmd_z = 0.0
        self.hight = self._nominal_height
        self.body_roll = 0.0
        self.body_pitch = 0.0
        self.body_yaw = 0.0
        self._clamp_commands()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for ManagerBasedRLEnv: {self.__class__.__name__}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        pass

    def advance(self):
        """Update torso-pose commands continuously for keys currently held down."""
        if not self._continuous_torso:
            return

        dt = float(self.env.step_dt)
        height_delta = self._height_rate * dt
        orientation_delta = self._orientation_rate * dt

        self.hight += height_delta * (("H" in self._held_torso_keys) - ("J" in self._held_torso_keys))
        self.body_roll += orientation_delta * (
            ("Z" in self._held_torso_keys) - ("X" in self._held_torso_keys)
        )
        self.body_pitch += orientation_delta * (
            ("C" in self._held_torso_keys) - ("V" in self._held_torso_keys)
        )
        self.body_yaw += orientation_delta * (
            ("B" in self._held_torso_keys) - ("N" in self._held_torso_keys)
        )
        self._clamp_commands()

    def _clamp_commands(self):
        for attr, (lo, hi) in self._cmd_limits.items():
            setattr(self, attr, float(max(lo, min(getattr(self, attr), hi))))

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # Isaac Sim may provide KeyboardInput on press and a plain string on release.
        key_name = getattr(event.input, "name", event.input)
        key_name = str(key_name).upper()
        if self._continuous_torso and key_name in self._TORSO_KEYS:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self._held_torso_keys.add(key_name)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._held_torso_keys.discard(key_name)
            return True

        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self._INPUT_KEY_MAPPING:
                if key_name == "R":
                    self.env.episode_length_buf = torch.ones_like(self.env.episode_length_buf) * 1e6
                    self._held_torso_keys.clear()
                    self.cmd_x = 0.0
                    self.cmd_y = 0.0
                    self.cmd_z = 0.0
                    self.hight = self._nominal_height
                    self.body_roll = 0.0
                    self.body_pitch = 0.0
                    self.body_yaw = 0.0
                if key_name == "W":
                    self.cmd_x += 0.1
                    self._clamp_commands()
                    print(f"Command X: {self.cmd_x}, Command Y: {self.cmd_y}, Command Z: {self.cmd_z}")
                if key_name == "S":
                    self.cmd_x -= 0.1
                    self._clamp_commands()
                    print(f"Command X: {self.cmd_x}, Command Y: {self.cmd_y}, Command Z: {self.cmd_z}")
                if key_name == "A":
                    self.cmd_y += 0.1
                    self._clamp_commands()
                    print(f"Command X: {self.cmd_x}, Command Y: {self.cmd_y}, Command Z: {self.cmd_z}")
                if key_name == "D":
                    self.cmd_y -= 0.1
                    self._clamp_commands()
                    print(f"Command X: {self.cmd_x}, Command Y: {self.cmd_y}, Command Z: {self.cmd_z}")
                if key_name == "Q":
                    self.cmd_z += 0.1
                    self._clamp_commands()
                    print(f"Command X: {self.cmd_x}, Command Y: {self.cmd_y}, Command Z: {self.cmd_z}")
                if key_name == "E":
                    self.cmd_z -= 0.1
                    self._clamp_commands()
                    print(f"Command X: {self.cmd_x}, Command Y: {self.cmd_y}, Command Z: {self.cmd_z}")
                if key_name == "H":
                    self.hight += 0.01
                    self._clamp_commands()
                    print(f"hignt:{self.hight}")
                if key_name == "J":
                    self.hight -= 0.01
                    self._clamp_commands()
                    print(f"hignt:{self.hight}")
                if key_name == "Z":
                    self.body_roll += 0.1
                    self._clamp_commands()
                    print(f"roll:{self.body_roll}")
                if key_name == "X":
                    self.body_roll -= 0.1
                    self._clamp_commands()
                    print(f"roll:{self.body_roll}")
                if key_name == "C":
                    self.body_pitch += 0.1
                    self._clamp_commands()
                    print(f"pitch:{self.body_pitch}")
                if key_name == "V":
                    self.body_pitch -= 0.1
                    self._clamp_commands()
                    print(f"pitch:{self.body_pitch}")
                if key_name == "B":
                    self.body_yaw += 0.1
                    self._clamp_commands()
                    print(f"yaw:{self.body_yaw}")
                if key_name == "N":
                    self.body_yaw -= 0.1
                    self._clamp_commands()
                    print(f"yaw:{self.body_yaw}")

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # forward command
            "R": "reset envs",
            "W": "move forward",
            "S": "move backward",
            "A": "move left",
            "D": "move right",
            "Q": "move up",
            "E": "move down",
            "H": "sdsd",
            "J": "sdsd",
            "Z": "sdsd",
            "X": "asas",
            "C": "asas",
            "V": "asas",
            "B": "asas",
            "N": "asas",
        }
