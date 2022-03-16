import torch
import json
import os
from torch import nn

def _get_channel_feature(img, res):
    features_list = helper.extract_features_fn(img)
    for l, feature in enumerate(features_list):
        feature = feature.squeeze()
        feature = feature.view(feature.shape[0], -1)
        mean = feature.mean(-1)
        res[l] += [mean.tolist()]
    return res


class Helper():
    def __init__(self, model):
        self.extract_features = []
        self.feature_level_num = len(self.extract_features_fn(torch.rand(1, 3, 512, 512).cuda()))

    def forward(self, img, mask=None, textbox=False):
        self.extract_features = []
        #img = nn.functional.interpolate(img, (224, 224))
        if textbox:
            outputs = self.model(img.float().cuda(), attack=True)
        else:
            outputs = self.model(img.float().cuda())


        features = outputs[1]
        text_features, nottext_features = [], []
        if mask is None:
            self.extract_features = []
            return outputs, features, text_features, nottext_features

        for feature in features:
            #if feature.shape[-1]!=img.shape[-1]//8: continue

            if len(mask.shape)==3:
                mask_ = nn.functional.interpolate(mask.unsqueeze(1), feature.shape[-2:])
            else:
                mask_ = nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), feature.shape[-2:])
            mask_ = mask_.reshape(-1)!=0
            feature = feature.squeeze().permute(1,2,0)
            feature = feature.view(-1, feature.shape[-1])
            
            text_feature = feature[mask_]
            nontext_feature = feature[mask_==0]

            text_features += [text_feature]
            nottext_features += [nontext_feature]

        self.extract_features = []
        #import pdb; pdb.set_trace()
        return outputs, features, text_features, nottext_features

    def get_original_features(self, model,  dataset, textbox=False):
        for idx in range(len(dataset)):
            item = dataset.__getitem__(idx)
            img_path = item['filename']
            mask = item['mask'].cuda()
            img = model.load_image(img_path)

            _, _, text_features, nontext_features = self.forward(img, mask, textbox=textbox)
            if idx==0:
                text_feature_len, nontext_feature_len = [0]*len(text_features), [0]*len(text_features)
                text_feature_mean, nontext_feature_mean = [0]*len(text_features), [0]*len(text_features)
            for i in range(len(text_features)):



                text_feature_mean[i] = ((text_feature_mean[i] * text_feature_len[i]) + \
                        text_features[i].sum(0).detach()) / (text_feature_len[i]+len(text_features[i]))
                nontext_feature_mean[i] = ((nontext_feature_mean[i] * nontext_feature_len[i]) + \
                        nontext_features[i].sum(0).detach()) / (nontext_feature_len[i]+len(nontext_features[i]))
                text_feature_mean[i] = torch.rand(text_feature_mean[i].shape, device=text_feature_mean[i].device)
                nontext_feature_mean[i] = torch.rand(nontext_feature_mean[i].shape, device=nontext_feature_mean[i].device)

                text_feature_len[i]+=len(text_features[i])
                nontext_feature_len[i]+=len(nontext_features[i])
            break
        self.original_text_features = text_feature_mean
        self.original_nontext_features = nontext_feature_mean

    def loss(self, text_features, nontext_features):
        feature_loss = 0
        def chamfer_loss(text_feature, original_text_feature):
            text_feature, original_text_feature = text_feature, original_text_feature
            distance_a2b = (text_feature[:,None, :] * original_text_feature[None, :,:]).sum(-1).min(-1)[0].mean()
            distance_b2a = (original_text_feature[:,None, :] * text_feature[None, :,:]).sum(-1).min(-1)[0].mean()
            return (distance_a2b + distance_b2a).cuda()


        len_ = len(text_features)
        weights = [(1/2)**abs(i-(len_//2)) for i in range(len_) ]


        for i, (text_feature, nontext_feature) in enumerate(zip(text_features, nontext_features)):
            if len(text_feature)==0: continue


            original_text_feature, original_nontext_feature = self.original_text_features[i], self.original_nontext_features[i]

            #original_text_feature, original_nontext_feature = original_text_feature.mean(0), original_nontext_feature.mean(0)
            # chamfer loss
            #loss += chamfer_loss(text_feature, original_text_feature)


            text_feature = text_feature / torch.sqrt((text_feature**2).sum(-1))[:, None]
            original_nontext_feature = original_nontext_feature / torch.sqrt((original_nontext_feature**2).sum())
            original_text_feature = original_text_feature / torch.sqrt((original_text_feature**2).sum())
            #loss_ = text_feature.abs().sum(-1).mean()

            loss_ = -(1-(text_feature * original_text_feature[None, :]).sum(-1).mean())
            loss_ += 1-(text_feature * original_nontext_feature[None, :]).sum(-1).mean()
            feature_loss += loss_*weights[i]


            #loss += [text_feature.abs().sum()]
        return feature_loss
            
    def extract_features_fn(self, image):
        """
        image is a tensor, with shape b x 3 x H x W
        """
        self.extract_features = []
        logits = self.model(image) # b x d
        features = self.extract_features
        self.extract_features = []
        return features

    def get_means_std(self, images):
        ## the images belong to the same label, that is the target label;

        res = [[] for _ in range(self.feature_level_num)]
        for img in images: res = _get_channel_feature(img, res)

        means, stds = [], []

        for mean in res:
            mean = torch.tensor(mean, dtype=torch.float, device="cuda") # n x d

            stds += [torch.from_numpy(
                np.cov(mean.transpose(1, 0))).to("cuda")]
            means += [mean.mean(dim=0)]

        return  means, stds

class ResnetHelper(Helper):
    def __init__(self, model):
        self.model = model
        self.forward_rule_count = 0
#        self.register_hooks()
        super().__init__(model)

    def register_hooks(self):
        return 
        def forward_hook_fn(module, inputs, outputs):
            self.forward_rule_count += 1
            if not self.forward_rule_count % 3 == 0: return
            inputs = inputs[0]
            self.extract_features += [inputs]

        def model_forward_hook_fn(model, inputs, outputs):
            self.forward_rule_count = 0

        def backward_hook_fn(module, grad_in, grad_out):
            pass

        def replace_relu(model):
            for name, module in self.model.named_modules():
                if "Bottleneck" in str(module)[:10]:
                     for child_name, child in module.named_children():
                         if child_name == "relu":
                             setattr(module, child_name, nn.ReLU())
                             module._modules[child_name].register_forward_hook(forward_hook_fn)

        ## change relu(inplace=True) to relu
        replace_relu(self.model)
        self.model.register_forward_hook(model_forward_hook_fn)

class  DensenetHelper(Helper):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.register_hooks()

    def register_hooks(self):
        def forward_hook_fn(module, inputs, outputs):
            self.extract_features += [outputs]

        def model_forward_hook_fn(model, inputs, outputs):
            self.forward_rule_count = 0
            
        for name, module in self.model.named_modules():
            if "_DenseBlock" in str(module)[:11] or "_Transition" in str(module)[:11]:
                module.register_forward_hook(forward_hook_fn)

        self.model.register_forward_hook(model_forward_hook_fn)

class VGGHelper(Helper):
    def __init__(self, model):
        #super().__init__(model)
        self.model = model
        self.register_hooks()
        super().__init__(model)

    def register_hooks(self):
        return
        def forward_hook_fn(module, inputs, outputs):
            self.extract_features += [outputs]

        def replace_relu(model):
            count = 0
            for child_name, child in model.named_children():
                if isinstance(child, nn.ReLU):
                #if isinstance(child, nn.MaxPool2d):
                    setattr(model, child_name, nn.ReLU())
                    model._modules[child_name].register_forward_hook(forward_hook_fn)
                else: replace_relu(child)

        replace_relu(self.model)

class InceptionHelper(Helper):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.register_hooks()


    def register_hooks(self):
        def forward_hook_fn(module, inputs, outputs):
            pass
        pass
