"""
Usage:
python load_policy.py -i /home/yuhsienc/workspace/cup_wild_vit_l_1img.ckpt
"""
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import click
import pathlib
import torch
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
@click.command()
@click.option('-i', '--input', required=True)
def main(input):

    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # creating model
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda:0')
    policy.eval().to(device)

    # set inference params
    # policy.num_inference_steps = 16   # DDIM inference iterations
    # policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    # n_obs_steps = policy.n_obs_steps
    print('sleep 10s')
    import time
    time.sleep(10)

if __name__ == "__main__":
    main()
