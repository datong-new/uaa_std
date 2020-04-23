#!python3/
from collections import OrderedDict
import json
import sys
sys.path.insert(0,"/data/text_attack/advGAN_pytorch/DB")
import argparse
import os
import torch
import yaml
from tqdm import tqdm
import numpy as np
from trainer import Trainer
# tagged yaml objects
from experiment import Structure, TrainSettings, ValidationSettings, Experiment
from concern.log import Logger
from data.data_loader import DataLoader
from data.image_dataset import ImageDataset
from training.checkpoint import Checkpoint
from training.learning_rate import (
    ConstantLearningRate, PriorityLearningRate, FileMonitorLearningRate
)
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from concern.config import Configurable, Config
import time

class Model:
    def __init__(self, cmd=dict(), verbose=False):
        os.chdir("/data/text_attack/advGAN_pytorch/DB")
        with open("db_args.json", "r") as f:
            args = json.load(f)
        conf = Config()
        experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
        experiment_args.update(cmd=args)
        experiment = Configurable.construct_class_from_config(experiment_args)
        cmd = args
        verbose = args['verbose']
        args = experiment_args

        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.data_loaders = experiment.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment.logger
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = cmd.get(
            'resume', os.path.join(
                self.logger.save_dir(model_saver.dir_path),
                'final'))
        self.verbose = verbose
        self.init_model()
        self.resume(self.net, self.model_path)

#        self.model.eval()
        
    def init_torch_tensor(self):
        # Use gpu or not
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        self.net = self.structure.builder.build_basic_model().cuda()
        
        return self.net

    def resume(self, net, path):
        if not os.path.exists(path):
            self.logger.warning("Checkpoint not found: " + path)
            return
        self.logger.info("Resuming from " + path)
        states = torch.load(
            path, map_location='cpu')

        new_state_dict = OrderedDict()
        for k, v in states.items():
            name = k[13:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.net.load_state_dict(new_state_dict)
        self.logger.info("Resumed from " + path)

    def report_speed(self, net, batch, times=100):
        data = {k: v[0:1]for k, v in batch.items()}
        if  torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time() 
        for _ in range(times):
            pred = self.net.forward(data)
        for _ in range(times):
            output = self.structure.representer.represent(batch, pred, is_output_polygon=False) 
        time_cost = (time.time() - start) / times
        self.logger.info('Params: %s, Inference speed: %fms, FPS: %f' % (
            str(sum(p.numel() for p in net.parameters() if p.requires_grad)),
            time_cost * 1000, 1 / time_cost))
        
        return time_cost
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        
    def eval(self, visualize=False):
        os.chdir("/data/text_attack/advGAN_pytorch/DB")
        all_matircs = {}
        self.net.eval()
        vis_images = dict()
        with torch.no_grad():
            for _, data_loader in self.data_loaders.items():
                raw_metrics = []
                for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    if self.args['test_speed']:
                        time_cost = self.report_speed(self.net, batch, times=50)
                        continue
                    pred, features = self.net.forward(batch, training=False)
                    output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
                    if not os.path.isdir(self.args['result_dir']):
                        os.mkdir(self.args['result_dir'])
                    self.format_output(batch, output)
                    raw_metric = self.structure.measurer.validate_measure(batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                    raw_metrics.append(raw_metric)

                    if visualize and self.structure.visualizer:
                        vis_image = self.structure.visualizer.visualize(batch, output, pred)
                        self.logger.save_image_dict(vis_image)
                        vis_images.update(vis_image)
                metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
                for key, metric in metrics.items():
                    self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))

    def get_features(self, x):
        self.net.eval()
        pred, features = self.net.forward(x, training=False)
        return features
