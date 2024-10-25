"""

"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import pathlib
import numpy as np
import cv2
import pickle
import time
from umi.common.cv_util import draw_predefined_mask
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.uvc_camera import UvcCamera
from umi.real_world.video_recorder import VideoRecorder
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import get_image_transform
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from functools import partial


def tf(data, input_res, output_res, fisheye_converter=None, is_mirror=None, no_mirror=True):
    """
    """
    img = data['color']
    if fisheye_converter is None:
        f = get_image_transform(
            input_res=input_res,
            output_res=output_res,
            # obs output rgb
            bgr_to_rgb=True)
        img = np.ascontiguousarray(f(img))
        if is_mirror is not None:
            img[is_mirror] = img[:,::-1,:][is_mirror]

        mask_image = np.zeros((output_res[1], output_res[0], 3), dtype=np.uint8)
        mask_image = draw_predefined_mask(mask_image, color=(255,255,255), 
            mirror=no_mirror, gripper=False, finger=False, use_aa=True)
        img = (img * 0.6 + img * (mask_image/255.0) * 0.4).astype(np.uint8)
    else:
        img = fisheye_converter.forward(img)
        img = img[...,::-1]
    data['color'] = img
    return data


def vis_tf(data, input_res, vis_res, masked_vis=True, no_mirror=True):
    """
    """
    img = data['color']
    f = get_image_transform(
        input_res=input_res,
        output_res=vis_res,
        bgr_to_rgb=False
    )
    img = f(img)

    if masked_vis:
        mask_image = np.zeros((vis_res[1], vis_res[0], 3), dtype=np.uint8)
        mask_image = draw_predefined_mask(mask_image, color=(255,255,255), 
            mirror=no_mirror, gripper=True, finger=False, use_aa=True)
        img = (img * 0.6 + img * (mask_image/255.0) * 0.4).astype(np.uint8)

    data['color'] = img
    return data


# %%
@click.command()
@click.option('-o', '--output', default='../saved_image.jpg', required=True, type=str)
@click.option('-mp', '--mask_write_path', default='../slam_mask.png', help='Mask file')
@click.option('-h', '--height', required=False, default=1080, help='Image height')
@click.option('-w', '--width', required=False, default=1920, help='Image width')
@click.option('-fr', '--fps', required=False, default=60, help='fps')
@click.option('--vis_camera_idx', default=0, type=int)
def main(output, mask_write_path, height, width, fps, vis_camera_idx):

    vis_res = (1280, 720)  # w,h

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

    video_recorder = VideoRecorder.create_hevc_nvenc(
            fps=fps,
            input_pix_fmt='bgr24',
            bit_rate=3000*1000
        )

    partial_tf = partial(tf, input_res=(width, height), output_res=(width, height), fisheye_converter=None, is_mirror=None, no_mirror=True)
    partial_vis_tf = partial(vis_tf, input_res=(width, height), vis_res=vis_res)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter,\
            UvcCamera(
                dev_video_path=v4l_paths[vis_camera_idx],
                shm_manager=shm_manager,
                resolution=(width, height),
                capture_fps=fps,
                # send every frame immediately after arrival
                # ignores put_fps
                put_downsample=False,
                get_max_k=60,
                receive_latency=0.125,
                cap_buffer_size=1,
                transform=partial_tf,
                vis_transform=partial_vis_tf,
                video_recorder=video_recorder,
                verbose=False) as camera:

            # warm up GUI
            data = camera.get()
            img = data['color']
            cv2.imshow('frame', img)
            cv2.pollKey()
            record_data_buffer = list()

            # handle key presses
            stop = False
            while not stop:
                # handle image stuff
                data = camera.get()
                img = data['color']

                # Display the resulting frame
                vis = camera.get_vis()
                vis_img = vis['color']

                text = f'Num frames saved: {len(record_data_buffer)}'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )
                cv2.imshow('frame', vis_img)
                cv2.pollKey()

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                        
                    elif key_stroke == Key.space:
                            # record image
                            # record_data_buffer.append({
                            #     'img': img[...,::-1]
                            # })
                            out_path = pathlib.Path(output)
                            cv2.imwrite(out_path , img[...,::-1])
                            print('Image saved to buffer.')
                    # elif key_stroke == Key.backspace:
                    #         # delete latest
                    #         if len(record_data_buffer) > 0:
                    #             record_data_buffer.pop()
                    # elif key_stroke == KeyCode(char='s'):
                    #     # save
                    #     out_path = pathlib.Path(output)
                    #     out_path.parent.mkdir(parents=True, exist_ok=True)
                    #     pickle.dump(record_data_buffer, file=out_path.open('wb'))
                    #     print(f"Saved data to {output}")


# %%
if __name__ == '__main__':
    main()