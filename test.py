from train import Trainer
from utils.config import opt
import torch
from torch.utils.data import DataLoader
from data.fetus_dataset_coco import COCO_Dataset, collate_fn
import argparse

parser = argparse.ArgumentParser()
opt.local_rank = opt.enable_GPUs_id[0]
opt.distributed = False
opt.device = torch.device("cuda:{}".format(opt.local_rank))

parser.add_argument("--path", type=str, default='checkpoints/res101pre_gpu7_heart_c1-c2_05131829_0.6961499205725439')

args = parser.parse_args()
testset  = COCO_Dataset(opt, operation='test', domain='Target')
annnotations = testset.data['categories']
test_dataloader = DataLoader(testset,
                            collate_fn = collate_fn(opt),
                            batch_size=1,
                            num_workers=opt.test_num_workers,
                            shuffle=False,)
Train = Trainer(opt)
Train.load(args.path)
test_result = Train.eval(test_dataloader, test_num=opt.test_num)

print(f"mAP:{str(test_result['map'])}")
for i in range(1, len(test_result['ap'])):
    name = annnotations[i-1]['name']
    print(f"{name}--{round(test_result['ap'][i], 4)}")