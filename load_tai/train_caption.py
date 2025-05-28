import warnings
warnings.filterwarnings("ignore")

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

class Args:
    def __init__(self):
        self.config_file = "/media/chaod/code/TaI-DPT/configs/trainers/Caption_distill_double/rn50_coco2014.yaml"
        self.dataset_config_file = "/media/chaod/code/TaI-DPT/configs/datasets/cifar100_distill.yaml"
        self.model_dir = "/media/chaod/code/TaI-DPT/output/cifar100_10_caption/Caption_distill_double/rn50_coco2014/nctx16_cscFalse_ctpend/seed4"
        self.load_epoch = 3
        self.opts = None 
        self.seed = -1

def reset_cfg(cfg, args):

    if args.seed > -1:
        cfg.SEED = args.seed


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.Caption = CN()
    cfg.TRAINER.Caption.N_CTX = 16  # number of context vectors
    cfg.TRAINER.Caption.CSC = False  # class-specific context
    cfg.TRAINER.Caption.CTX_INIT = ""  # initialization words
    cfg.TRAINER.Caption.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.Caption.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.Caption.GL_merge_rate = 0.5
    # cfg.TRAINER.Caption.fewshot_TaI_merge_rate = 0.6
    # cfg.TRAINER.Caption.partial_TaI_merge_rate = 0.9

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SAMPLE = 0 # Sample some of all datas, 0 for no sampling i.e. using all
    cfg.DATASET.partial_prob = 0.5

    cfg.TRAIN.LOSSFUNC = ""  # or "focal"
    cfg.TRAIN.IF_LEARN_SCALE = False
    cfg.TRAIN.IF_LEARN_spatial_SCALE = False
    cfg.TRAIN.spatial_SCALE_text = 50
    cfg.TRAIN.spatial_SCALE_image = 50
    cfg.TRAIN.IF_ablation = False
    cfg.TRAIN.Caption_num = 0
    
    cfg.TEST.EVALUATOR_ACT = "softmax"  # or "sigmoid"
    cfg.TEST.SAVE_PREDS = ""
    
    # several param for spacific transform setting
    cfg.INPUT.random_resized_crop_scale = (0.8, 1.0)
    cfg.INPUT.cutout_proportion = 0.4
    cfg.INPUT.TRANSFORMS_TEST = ("resize", "center_crop", "normalize")


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        print('merge_from_file {}'.format(args.dataset_config_file))
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)


    cfg.freeze()

    return cfg


def main():
    args = Args()
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    return cfg 


    
