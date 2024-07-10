import argparse
import itertools
import data
from train_disyn import pretrain_proc_disyn_adv
from utils import *
import path_config


def main(args_dict):
    set_seed(args.seed)

    pretrain_set = ['drop_out', 'nums_critic', 'nums_recon']
    args_dict.update(
        {
            'model_save_path': os.path.join(path_config.result_path, f'Disyn_{args_dict["seed"]}', set_to_str(args_dict, pretrain_set))
        })

    os.makedirs(args_dict['model_save_path'], exist_ok=True)

    s_dataloader, t_dataloader = data.get_unlabeled_dataloaders()
    args_dict.update({'input_dim': t_dataloader.dataset.tensors[0].shape[1]})

    # start unlabeled pretraining
    pretrain_proc_disyn_adv(s_dataloader, t_dataloader, **args_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=95, help='seed')
    parser.add_argument('--nums_recon', type=int, default=100, help='recon epochs in model pretrain')
    parser.add_argument('--nums_critic', type=int, default=100, help='critic epochs in model pretrain')
    parser.add_argument('--drop_out', type=float, default=0.0, help='drop_out')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    args = parser.parse_args()

    main(args_dict=vars(args))
