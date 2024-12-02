import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import logging


class DummyGripper:
    def __init__(self, inversed_pos=False):
        self._min_position = 0.  # stroke: 0 mm - 50 mm
        self._max_position = 60.
        self._min_speed = 20.  # speed: 20 mm/s - 150 mm/s
        self._max_speed = 150.
        self._min_force = 20.  # Grip force: 20 N - 185 N
        self._max_force = 185.

        self._current_position = 0.
        self._current_speed = 0.
        self._current_force = 0.

        self.inversed_pos = inversed_pos

    def get_current_position(self):
        return (self._max_position - self._current_position) if self.inversed_pos else self._current_position

    def get_current_speed(self):
        return self._current_speed

    def get_current_force(self):
        return self._current_force

    def reset(self):
        if self.inversed_pos:
            self._current_position = self._max_position
        else:
            self._current_position = self._min_position
        self._current_speed = 0.
        self._current_force = 0.

    def move(self, position: float, speed: float, force: float):
        if self.inversed_pos:
            self._current_position = self._max_position - position
        else:
            self._current_position = position
        self._current_speed = speed
        self._current_force = force

    def get_open_position(self) -> float:
        if self.inversed_pos:
            return self._max_position
        else:
            return self._min_position

    def get_close_position(self) -> float:
        if self.inversed_pos:
            return self._min_position
        else:
            return self._max_position


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class DummyGripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            hostname,
            port,
            frequency=10,
            home_to_open=True,
            move_max_speed=150.0,  # mm/s
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.1,
            use_meters=False,
            verbose=False
            ):
        """
        Args
            shm_manager (`SharedMemoryManager`):
            hostname (str):
            port (uint):
            frequency (uint):
            home_to_open (bool):
            move_max_speed (float):
            get_max_k (uint):
            command_queue_size (uint):
            launch_timeout (uint):
            receive_latency (float):
            use_meters (bool): Set if unit of input `pos` in the command method (e.g. `schedule_waypoint`) is meter. (default: False)
            verbose (bool): (default: False)
        """    
        super().__init__(name="DummyGripperController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed  # must be in mm/s
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # build ring buffer
        example = {
            #'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[DummyGripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        """
        Args
            pos (float): A waypoint to be scheduled. The unit of `pos`
            target_time (float):
        """
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        try:
            # start connection
            gripper = DummyGripper(inversed_pos=True)

            # home gripper to initialize
            # (stop immediately)(homing - gripper open)
            print("[DummyGripperController] Activating gripper...")
            gripper.reset()
            if self.home_to_open:
                open_pos = gripper.get_open_position()
                gripper.move(open_pos, 150., 20.)  # 0 mm, open position for Hand-E
            else:
                close_pos = gripper.get_close_position()
                gripper.move(close_pos, 150., 20.)  # 60 mm, close position for Hand-E
            
            # get initial position
            curr_pos = gripper.get_current_position() # DO NOT scale `curr_pos` since unit of `pose_interp` is mm at all.
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos,0,0,0,0,0]]
            )

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                gripper.move(target_pos, target_vel, 20)

                # get state
                state = {
                    'gripper_position': gripper.get_current_position() / self.scale,  # mm or m
                    'gripper_velocity': gripper.get_current_speed() / self.scale,  # mm or m
                    'gripper_force': gripper.get_current_force(),  # N
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos'] * self.scale  # m -> mm if use_meters
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                            t_start = command['target_time'] - time.time() + time.monotonic()
                            iter_idx = 1
                    else:
                        keep_running = False
                        break

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[DummyGripperController] Disconnected from robot: {self.hostname}")







