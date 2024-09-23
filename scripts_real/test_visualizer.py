# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import pathlib
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode, Key
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.cv_util import draw_predefined_mask


RES = (1920, 1080)
FPS = 60
# RES = (1280, 720)
# FPS = 10


def tf(data, input_res=RES, fisheye_converter=None, is_mirror=None, no_mirror=False, obs_float32=False):
    img = data['color']
    if fisheye_converter is None:
        f = get_image_transform(
            input_res=input_res,
            output_res=(224,224), 
            # obs output rgb
            bgr_to_rgb=True)
        img = np.ascontiguousarray(f(img))
        if is_mirror is not None:
            img[is_mirror] = img[:,::-1,:][is_mirror]
        img = draw_predefined_mask(img, color=(0,0,0), 
            mirror=no_mirror, gripper=True, finger=False, use_aa=True)
    else:
        img = fisheye_converter.forward(img)
        img = img[...,::-1]
    if obs_float32:
        img = img.astype(np.float32) / 255
    data['color'] = img
    return data


def vis_tf(data, input_res=RES, rw=960, rh=720):
    img = data['color']
    f = get_image_transform(
        input_res=input_res,
        output_res=(rw,rh),
        bgr_to_rgb=False
    )
    img = f(img)
    data['color'] = img
    return data


# %%
@click.command()
def main():

    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()

    # Wait for all v4l cameras to be back online
    time.sleep(0.1)
    v4l_paths = get_sorted_v4l_paths()

    # pop non-relevant paths
    for i, p in enumerate(v4l_paths):
        if 'Elgato' not in p:
            print(v4l_paths.pop(i))
    print(v4l_paths)

    # compute resolution for vis
    rw, rh, col, row = optimal_row_cols(
        n_cameras=len(v4l_paths),
        in_wh_ratio=4/3,
        max_resolution=(960, 960)
    )

    video_recorder = VideoRecorder.create_hevc_nvenc(
            fps=FPS,
            input_pix_fmt='bgr24',
            bit_rate=3000*1000
        )

    with KeystrokeCounter() as key_counter:

        shm_manager = SharedMemoryManager()
        shm_manager.start()

        camera = MultiUvcCamera(
                dev_video_paths=v4l_paths,
                shm_manager=shm_manager,
                resolution=[RES,],
                capture_fps=[FPS,],
                # send every frame immediately after arrival
                # ignores put_fps
                put_downsample=False,
                get_max_k=60,
                receive_latency=0.125,
                cap_buffer_size=[1,],
                transform=[tf,],
                vis_transform=[vis_tf,],
                video_recorder=[video_recorder,],
                verbose=False
            )

        multi_cam_vis = MultiCameraVisualizer(
                    camera=camera,
                    row=row,
                    col=col,
                    rgb_to_bgr=True
                )

        camera.start(wait=False)
        multi_cam_vis.start(wait=False)

        # handle key presses
        stop = False
        while not stop:
            press_events = key_counter.get_press_events()
            for key_stroke in press_events:
                if key_stroke == KeyCode(char='q'):
                    stop = True
                    camera.stop()
                    multi_cam_vis.stop()
                    shm_manager.shutdown()



# %%
if __name__ == '__main__':
    main()


