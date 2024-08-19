"""Module to control Robotiq's grippers - tested with HAND-E"""

import socket
import threading
import time
from enum import Enum
from typing import Union, Tuple, OrderedDict


def clip_val(val, range):
    min_val, max_val = range
    return max(min_val, min(val, max_val))

def norm_val(val, range) -> int:
    """Rescale the value within a specific range to [0,255], return as an integer. """
    val = clip_val(val, range)
    min_val, max_val = range
    ret = round((val_val-min_val)*255/(max_val-min_val), 0)
    return int(ret)

def unnorm_val(val, range) -> float:
    """ """
    val = clip_val(val, range)
    min_val, max_val = range
    ret = val/255.*(max_val-min_val)+min_val
    return round(ret, 3)


class RobotiqGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """
    # WRITE VARIABLES (CAN ALSO READ)
    ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = 'ATR'  # atr : auto-release (emergency slow move)
    ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = 'FOR'  # for : force (0-255)
    SPE = 'SPE'  # spe : speed (0-255)
    POS = 'POS'  # pos : position (0-255), 0 = open
    # READ VARIABLES
    STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = 'PRE'  # position request (echo of last commanded position)
    OBJ = 'OBJ'  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
    FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)

    ENCODING = 'UTF-8'  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""
        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self, hostname: str, port: int, socket_timeout: float = 2.0):
        """Constructor."""
        self.hostname = hostname
        self.port = port
        self.socket_timeout = socket_timeout
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0  # stroke: 0 mm - 50 mm
        self._max_position = 50
        self._min_speed = 20  # speed: 20 mm/s - 150 mm/s
        self._max_speed = 150
        self._min_force = 20  # Grip force: 20 N - 185 N
        self._max_force = 185

    def connect(self) -> None:
        """Connects to a gripper at the given address.
        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.hostname, self.port))
        self.socket.settimeout(self.socket_timeout)

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ================= low level API ================

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]):
        """Sends the appropriate command via socket to set the value of n variables, and waits for its 'ack' response.
        :param var_dict: Dictionary of variables to set (variable_name, value).
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        # construct unique command
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += '\n'  # new line is required for the command to finish
        # atomic commands send/rcv
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return self._is_ack(data)

    def _set_var(self, variable: str, value: Union[int, float]):
        """Sends the appropriate command via socket to set the value of a variable, and waits for its 'ack' response.
        :param variable: Variable to set.
        :param value: Value to set for the variable.
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """Sends the appropriate command to retrieve the value of a variable from the gripper, blocking until the
        response is received or the socket times out.
        :param variable: Name of the variable to retrieve.
        :return: Value of the variable as integer.
        """
        # atomic commands send/rcv
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)

        # expect data of the form 'VAR x', where VAR is an echo of the variable name, and X the value
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data} ({data.decode(self.ENCODING)}): does not match '{variable}'")
        value = int(value_str)
        return value

    @staticmethod
    def _is_ack(data: str):
        return data == b'ack'

    def _reset(self):
        """
        Reset the gripper.
        The following code is executed in the corresponding script function
        def rq_reset(gripper_socket="1"):
            rq_set_var("ACT", 0, gripper_socket)
            rq_set_var("ATR", 0, gripper_socket)

            while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                rq_set_var("ACT", 0, gripper_socket)
                rq_set_var("ATR", 0, gripper_socket)
                sync()
            end

            sleep(0.5)
        end
        """
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)

    # =============== high level API ===============

    def activate(self, auto_calibrate: bool = True):
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.
        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.
        The following code is executed in the corresponding script function
        def rq_activate(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_reset(gripper_socket)

                while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                    rq_reset(gripper_socket)
                    sync()
                end

                rq_set_var("ACT",1, gripper_socket)
            end
        end
        def rq_activate_and_wait(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_activate(gripper_socket)
                sleep(1.0)

                while(not rq_get_var("ACT", 1, gripper_socket) == 1 or not rq_get_var("STA", 1, gripper_socket) == 3):
                    sleep(0.1)
                end

                sleep(0.5)
            end
        end
        """
        if not self.is_active():
            self._reset()
            while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
                time.sleep(0.01)

            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            while (not self._get_var(self.ACT) == 1 or not self._get_var(self.STA) == 3):
                time.sleep(0.01)

        # auto-calibrate position range if desired
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self):
        """Returns whether the gripper is active."""
        status = self._get_var(self.STA)
        return RobotiqGripper.GripperStatus(status) == RobotiqGripper.GripperStatus.ACTIVE

    def get_min_position(self) -> int:
        """Returns the minimum position the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> float:
        """Returns the current position as returned by the physical hardware."""
        return unnorm_val(self._get_var(self.POS), [self._min_position, self._max_position])

    def get_current_speed(self) -> float:
        """Returns the current velocity as returned by the physical hardware."""
        return unnorm_val(self._get_var(self.SPE), [self._min_speed, self._max_speed])

    def get_current_force(self) -> float:
        """Returns the current force as returned by the physical hardware."""
        return unnorm_val(self._get_var(self.FOR), [self._min_force, self._max_force])

    def get_position_requested(self) -> float:
        """position requested (echo of last commanded position) """
        return unnorm_val(self._get_var(self.PRE), [self._min_position, self._max_position])

    def get_gripper_status(self) -> int:
        """ """
        return self._get_var(self.STA)

    def get_object_status(self) -> int:
        """ """
        return self._get_var(self.OBJ)

    def get_fault_status(self) -> int:
        """ """
        return self._get_var(self.FLT)

    def auto_calibrate(self, log: bool = True) -> None:
        """Attempts to calibrate the open and closed positions, by slowly closing and opening the gripper.
        :param log: Whether to print the results to log.
        """
        cali_speed = unnorm(64, [self._min_speed, self._max_speed])
        cali_force = unnorm(1, [self._min_force, self._max_force])
        
        # first try to open in case we are holding an object
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), cali_speed, cali_force)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {str(status)}")

        # try to close as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_closed_position(), cali_speed, cali_force)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position <= self._max_position
        self._max_position = position

        # try to open as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), cali_speed, cali_force)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position >= self._min_position
        self._min_position = position

        if log:
            print(f"Gripper auto-calibrated to [{self.get_min_position()}, {self.get_max_position()}]")

    def move(self, position: float, speed: float, force: float) -> Tuple[bool, float]:
        """Sends commands to start moving towards the given position, with the specified speed and force.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """

        clip_pos = norm(position, [self._min_position, self._max_position])
        clip_spe = norm(speed, [self._min_speed, self._max_speed])
        clip_for = norm(force, [self._min_force, self._max_force])

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([(self.POS, clip_pos), (self.SPE, clip_spe), (self.FOR, clip_for), (self.GTO, 1)])
        return self._set_vars(var_dict), unnorm(clip_pos, [self._min_position, self._max_position])

    def move_and_wait_for_pos(self, position: float, speed: float, force: float) -> Tuple[float, ObjectStatus]:  # noqa
        """Sends commands to start moving towards the given position, with the specified speed and force, and
        then waits for the move to complete.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with an integer representing the last position returned by the gripper after it notified
        that the move had completed, a status indicating how the move ended (see ObjectStatus enum for details). Note
        that it is possible that the position was not reached, if an object was detected during motion.
        """
        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move.")

        # wait until the gripper acknowledges that it will try to go to the requested position
        while self.get_position_requested() != cmd_pos:
            time.sleep(0.001)

        # wait until not moving
        cur_obj = self._get_var(self.OBJ)
        while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
            cur_obj = self._get_var(self.OBJ)

        # report the actual position and the object status
        final_pos = self.get_current_position()
        final_obj = cur_obj
        return final_pos, RobotiqGripper.ObjectStatus(final_obj)