import os
import argparse
import numpy as np
import torch
from scipy import stats

from utils.common import setup_seed, get_num_cls, get_test_labels

from utils.detection_util import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip, get_id_scores_clip
# from utils.detection_place import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip, get_id_scores_clip
# from utils.detection_dtd import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip, get_id_scores_clip

from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
# sys.path.append(os.path.dirname(__file__))


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet10', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100', 'waterbird',
                                  'pet37', 'food101', 'car196', 'bird200', 'cifar10','cifar100'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type = int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=[
        'min-max','MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha', 'aod','apd','aneard','changed-softmax','max-min'], help='score options')
    # for Mahalanobis score
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ViT-B and 768 for ViT-L')
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = False, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 250, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    args2 = parser.parse_args()

    args2.n_cls = get_num_cls(args2)
    args2.log_directory = f"results/{args2.in_dataset}/{args2.score}/{args2.model}_{args2.CLIP_ckpt}_T_{args2.T}_ID_{args2.name}"
    os.makedirs(args2.log_directory, exist_ok=True)

    return args2

def main():
    args2 = process_args()
    setup_seed(args2.seed)
    log = setup_log(args2)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args2.gpu)

    net, preprocess = set_model_clip(args2)
    net.eval()

    if args2.in_dataset in ['ImageNet10']: 
        out_datasets = ['ImageNet20']
        # out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']
    elif args2.in_dataset in ['ImageNet20']: 
        out_datasets = ['ImageNet10']
        # out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']
    elif args2.in_dataset in [ 'ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37']:
        out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']
        # out_datasets = ['Places_bg']
    elif args2.in_dataset in ['cifar100']:
        out_datasets = ['cifar10','tinyimagenet','lsun']
    elif args2.in_dataset in ['cifar10']:
        out_datasets = ['cifar100','cifar100_easy','cifar100_hard','iNaturalist','SUN', 'places365', 'dtd','ImageNet10','ImageNet20']
    elif args2.in_dataset in ['waterbird']:
        out_datasets = ['Places_bg']
 
    test_loader = set_val_loader(args2, preprocess)
    test_labels = get_test_labels(args2, test_loader) #获取ID样本类别名
    if args2.score == 'maha':
        os.makedirs(args2.template_dir, exist_ok = True)
        train_loader = set_train_loader(args2, preprocess, subset = args2.subset) 
        if args2.generate: 
            classwise_mean, precision = get_mean_prec(args2, net, train_loader)
        classwise_mean = torch.load(os.path.join(args2.template_dir, f'{args2.model}_classwise_mean_{args2.in_dataset}_{args2.max_count}_{args2.normalize}.pt'), map_location= 'cpu').cuda()
        precision = torch.load(os.path.join(args2.template_dir,  f'{args2.model}_precision_{args2.in_dataset}_{args2.max_count}_{args2.normalize}.pt'), map_location= 'cpu').cuda()
        in_score = get_Mahalanobis_score(args2, net, test_loader, classwise_mean, precision, in_dist = True)
    else:
        in_score  = get_ood_scores_clip(args2, net, test_loader, test_labels, in_dist=True)
    
    if args2.score == 'changed-softmax' or args2.score == 'min-max' or args2.score == 'max-min':   
        in_score = np.array(in_score)
    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        # if out_dataset != 'iNaturalist':
            # continue
        # if out_dataset != 'SUN':
            # continue
        # if out_dataset != 'places365':
            # continue
        # if out_dataset != 'dtd':
            # continue
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        
        ood_loader = set_ood_loader_ImageNet(args2, out_dataset, preprocess, root=os.path.join(args2.root_dir, 'ImageNet_OOD_dataset'))
        print(len(ood_loader))
        if args2.score == 'maha':
            out_score = get_Mahalanobis_score(args2, net, ood_loader, classwise_mean, precision, in_dist = False)
        else:
            out_score = get_ood_scores_clip(args2, net, ood_loader, test_labels)
        
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        if args2.score == 'changed-softmax' or args2.score == 'min-max' or args2.score == 'max-min':  
            out_score = np.array(out_score)
        plot_distribution(args2, in_score, out_score, out_dataset)
        get_and_print_results(args2, log, in_score, out_score,
                              auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args2.score)
    save_as_dataframe(args2, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == '__main__':
    main()
