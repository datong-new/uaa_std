#CUDA_VISIBLE_DEVICES=3 python eval.py --model db
#CUDA_VISIBLE_DEVICES=4 python eval.py --model textbox
#CUDA_VISIBLE_DEVICES=5 python eval.py --model textbox
#CUDA_VISIBLE_DEVICES=6 python eval.py --model textbox


#exit




use_di=False
use_ti=False
use_feature_loss=True
model=craft
#CUDA_VISIBLE_DEVICES=2 python eval.py --attack_type universal  --use_di $use_di --use_ti $use_ti --use_feature_loss $use_feature_loss  --model $model --partial_eval True
CUDA_VISIBLE_DEVICES=2 python eval.py --attack_type single --use_di $use_di --use_ti $use_ti --use_feature_loss $use_feature_loss  --model $model --partial_eval True
