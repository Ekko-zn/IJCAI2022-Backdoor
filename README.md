This repo is initial implementation for the paper "[Imperceptible Backdoor Attack: From Input Space to Feature Representation]"

## How to run
Train GTSRB dataset
```
python main.py --dataset GTSRB --num_class 43 --a 0.3 --b 0.1 --weight_decay 0
```
Train CelebA dataset
```
python main.py --dataset CelebA --num_class 8 --a 0.3 --b 0.1 --weight_decay 1e-4
```

Evaluate well-trained models
```
python eval.py --dataset GTSRB --num_class 43
python eval.py --dataset CelebA --num_class 8
```

## Tips
Download data from Baidudisk(code:sft2) and unzip it to root folder.
https://pan.baidu.com/s/17PnRjznAvxnC4p_XZnZVrw 
Well-trained models are accessible from Baidudisk(code:ur1g)
https://pan.baidu.com/s/1Q5yyVBQ4EHJBm1ChjDSXPQ 
