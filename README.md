# Imperceptible Backdoor Attack: From Input Space to Feature Representation

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
Pytorch version: 1.10.1

Download data from Baidudisk(code:sft2)/Google driver and unzip it to root folder.  
https://pan.baidu.com/s/17PnRjznAvxnC4p_XZnZVrw  
https://drive.google.com/file/d/1o_T6VNS8FHu1EDvBKEjw92agbK1sD0p9/view?usp=sharing

Well-trained models are accessible from Baidudisk(code:ur1g)/Google driver.
https://pan.baidu.com/s/1Q5yyVBQ4EHJBm1ChjDSXPQ   
https://drive.google.com/file/d/1FZKTVREnITtAVKoo_Cx4Wonh9X6RtE_l/view?usp=sharing
