# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
import pickle
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.dummy_gripper import DummyGripperController
from umi.common.precise_sleep import precise_wait

# %%
@click.command()
@click.option('-o', '--output', default=None, type=str)
@click.option('-h', '--hostname', default='192.168.10.3')
@click.option('-p', '--port', type=int, default=63352)
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-ms', '--max_speed', type=float, default=150.0)
@click.option('-mp', '--max_pos', type=float, default=50.0)
def main(output, hostname, port, frequency, max_speed, max_pos):
    duration = 60.0
    get_max_k = int(duration * frequency)
    command_latency = 0.0
    dt = 1/frequency

    with SharedMemoryManager() as shm_manager:
        with DummyGripperController(
            shm_manager=shm_manager,
            hostname=hostname,
            port=port,
            frequency=frequency,
            move_max_speed=max_speed,
            get_max_k=get_max_k,
            verbose=True) as gripper:

            print('Ready!')
            target_pos_traj = list()
            target_timestamps = list()
            try:
                # to account for recever interfance latency, use target pose
                # to init buffer.
                state = gripper.get_state()
                target_pos = state['gripper_position']
                # target_pos = 100.0
                t_start = time.monotonic()
                gripper.restart_put(t_start-time.monotonic() + time.time())

                iter_idx = 0
                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    precise_wait(t_sample)
                    sm_state = np.zeros((6,))
                    dpos = sm_state[0] * max_speed / frequency  # sm_state[0] -> froward-backward translation, [-0.7,0.7]
                    target_pos = np.clip(target_pos + dpos, 0, max_pos)
                    #print(target_pos)
                    target_timestamp = t_command_target-time.monotonic()+time.time()

                    gripper.schedule_waypoint(target_pos, target_timestamp)
                    target_pos_traj.append(target_pos)
                    target_timestamps.append(target_timestamp)

                    precise_wait(t_cycle_end)
                    iter_idx += 1
            except KeyboardInterrupt:
                if output is not None:
                    robot_state = gripper.get_all_state()
                    result = {
                        'target_position': np.array(target_pos_traj),
                        'target_timestamp': np.array(target_timestamps),
                        'actual_position': robot_state['gripper_position'],
                        #'actual_measure_timestamp': robot_state['gripper_measure_timestamp'],
                        'actual_receive_timestamp': robot_state['gripper_receive_timestamp']
                    }
                    print(f'Saving results to {output}')
                    pickle.dump(result, open(output, 'wb'))  



# %%
if __name__ == '__main__':
    main()
