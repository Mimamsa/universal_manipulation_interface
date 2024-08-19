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
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.precise_sleep import precise_wait

# %%
@click.command()
@click.option('-f', '--frequency', type=float, default=30)
def main(frequency):
    duration = 60.0
    get_max_k = int(duration * frequency)
    command_latency = 0.0
    dt = 1/frequency

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm:
            print('Ready!')
            try:
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    sm_state = sm.get_motion_state_transformed()
                    print(sm_state)
                    iter_idx += 1
            except KeyboardInterrupt:
                print('Exit')


# %%
if __name__ == '__main__':
    main()
