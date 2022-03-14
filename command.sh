# transfer attack
CUDA_VISIBLE_DEVIACES=0 python transfer_attack.py --attack_type universal --model_name craft

# generate adversarial examples
CUDA_VISIBLE_DEVIACES=0 python craft.py --attack_type single

