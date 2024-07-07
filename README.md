# Training instruction for HQ-SAM

We organize the training folder as follows.
```
train
|____datasets
|____segment_anything_training
|____train_IRSAM.py
|____utils
| |____dataloader.py
| |____log.py
| |____misc.py
| |____metric.py
| |____metrics.py
| |____loss_mask.py
| |____misc.py
|____workdirs
|____mobile_sam.pt
```

According to the given train/test.txt, place the training/testing images/labels into four separate folders.

## 1. Training
To train IRSAM on various dataset, modify the dataset path in the train-IRSAM.py file

### Example 
```
python train_IRSAM.py --output workdirs/your_workdir --checkpoint mobile_sam.pt
```

## 4. Evaluation
To evaluate IRSAM on various dataset, modify the dataset path in the train-IRSAM.py file

### Example
```
python train_IRSAM.py --output workdirs/your_workdir --checkpoint your_checkpoint --eval 
```
