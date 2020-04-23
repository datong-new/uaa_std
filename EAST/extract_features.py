import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	features = detect_dataset(model, device, test_img_path, submit_path)


if __name__ == '__main__': 
	model_name = './pths/east_vgg16.pth'
	#test_img_path = os.path.abspath('../ICDAR_2015/test_img')
	test_img_path = os.path.abspath('./img_dir')
	submit_path = './submit'
	eval_model(model_name, test_img_path, submit_path)
