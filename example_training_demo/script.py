import numpy as np
import zarr
import cv2
from skvideo.io import vwrite
import matplotlib.pyplot as plt

for domain_id in [0, 7, 13, 15, 18, 20, 21, 22]:
    dataset_path = f"/home/jordan/Han/feature_analysis/01_18_2026_release/dataset_domain/domain{domain_id}/domain{domain_id}.zarr"

    dataset_root = zarr.open(dataset_path, 'r')

    demo_end = dataset_root['meta']['episode_ends'][:]

    for i in range(1):
        if i == 0:
            imgs = dataset_root['data']['img'][0:demo_end[i]]
        else:
            imgs = dataset_root['data']['img'][demo_end[i-1]:demo_end[i]]
            

        # Set video writing parameters
        outputdict = {
            '-vcodec': 'libx264',  # Use H.264 codec
            '-crf': '23',  # Constant Rate Factor (CRF) for quality (lower is better, 23 is default)
            '-preset': 'slow',  # Encoding preset (slower = better compression)
            '-movflags': '+faststart',  # Optimize for web streaming
            '-pix_fmt': 'yuv420p',  # Pixel format for better compatibility
        }

        inputdict = {
            '-r': '20'  # Set the frame rate
        }

        vwrite(f"domain_{domain_id}_example_video.mp4", imgs, inputdict=inputdict, outputdict=outputdict)