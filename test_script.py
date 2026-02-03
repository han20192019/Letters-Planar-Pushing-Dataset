import numpy as np
import argparse
import wandb
import os
import yaml


# other files
from utils import *
from pusht_env import *
from models import *
from eval_baseline import eval_baseline
def main():

    

    checkpoint_dir = "./domain13_trained_model/checkpoint_epoch_1"

    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/train_config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
   

    num_train_demos = config['num_train_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    dataset_path_dir = config['dataset_path_dir']
    display_name = config['display_name']
    resize_scale = config["resize_scale"]


    if display_name == "default":
        display_name = None
    if config["wandb"]:
        wandb.init(
            project="test",
            config=config,
            name=display_name,
            reinit=True
        )
    else:
        print("warning: wandb flag set to False")


    resize_scale = 96

   
    dataset_list = []
    combined_stats = []
    num_datasets = 0
    dataset_name = {} # mapping for domain filename

    for entry in sorted(os.listdir(dataset_path_dir)):
        if not (entry[-5:] == '.zarr'):
            continue
        full_path = os.path.join(dataset_path_dir, entry)

        domain_filename = entry.split(".")[0]
        dataset_name[num_datasets] = domain_filename        

        # create dataset from file
        dataset = PushTImageDataset(
            dataset_path=full_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            id = num_datasets,
            num_demos = num_train_demos,
            resize_scale = resize_scale
        )
        num_datasets += 1
        # save training data statistics (min, max) for each dim
        stats = dataset.stats
        dataset_list.append(dataset)
        combined_stats.append(stats)



    scores = eval_baseline(config, dataset_name, num_datasets, combined_stats, checkpoint_dir)


    print(scores)

if __name__ == "__main__":
    main()