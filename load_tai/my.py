import train_caption
from Caption_distill_double import DenseCLIP,load_clip_to_cpu
import torch
from dassl.utils import load_pretrained_weights
from PIL import Image
import torchvision.transforms as transforms

cfg = train_caption.main()
device = torch.device('cuda')
print("Building model")

clip_model = load_clip_to_cpu(cfg)

classname = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', \
 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', \
    'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', \
        'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', \
            'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', \
                'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',\
                      'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm', 'horse', 'deer', 'peach', 'cat', 'dog', 'truck', 'gorillas', 'tortoise', 'llama', 'object hard to identify']

model = DenseCLIP(cfg, classname , clip_model)
load_pretrained_weights(model.prompt_learner, "/media/chaod/code/TaI-DPT/output/cifar100_10_caption/Caption_distill_double/rn50_coco2014/nctx16_cscFalse_ctpend/seed4/prompt_learner/model-best.pth.tar")
model.to(device)

img = Image.open('/media/chaod/code/TaI-DPT/data/cifar100/test/102/3.png')

normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    # normalize
])

tensor = preprocess(img)
output_, _,_,_ =  model(image= tensor.unsqueeze(0).to(device),captions=None,if_test=True)
print(output_)
values, indices = output_.max(dim=1)
print(values)
print(indices)
