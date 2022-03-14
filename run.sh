use_di=False
use_ti=False
use_feature_loss=True
model=textbox

CUDA_VISIBLE_DEVICES=1 python attacks.py --attack_type single --eps 13 --use_di $use_di --use_ti $use_ti --use_feature_loss $use_feature_loss  --model $model
