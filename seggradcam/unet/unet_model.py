""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import numpy as np
import cv2

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, start_filters, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, start_filters)
        self.down1 = Down(start_filters, start_filters*2)
        self.down2 = Down(start_filters*2, start_filters*4)
        self.down3 = Down(start_filters*4, start_filters*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(start_filters*8, start_filters*16 // factor)
        self.up1 = Up(start_filters*16, start_filters*8 // factor, bilinear)
        self.up2 = Up(start_filters*8, start_filters*4 // factor, bilinear)
        self.up3 = Up(start_filters*4, start_filters*2 // factor, bilinear)
        self.up4 = Up(start_filters*2, start_filters, bilinear)
        self.outc = OutConv(start_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):  # When deriving x, save the gradient
        outputs = []
        self.gradients = []
        multi_out = [x]
        for name, module in self.model._modules.items():
            print(name)
            if "up1" in name:
                x = module(x, multi_out[-2])
            elif "up2" in name:
                x = module(x, multi_out[-3])
            elif "up3" in name:
                x = module(x, multi_out[-4])
            elif "up4" in name:
                x = module(x, multi_out[-5])
            else:
                x = module(x)
                multi_out.append(x)

            if name in self.target_layers:
                print("Found target layer!")
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x




class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []

        y = self.model(x)
        #print(x.shape)
        target_activations, x = self.feature_extractor(x)

        return target_activations, torch.sigmoid(x)



class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda=True):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())  # [1, 2048, 7, 7], [1, 1000]
        else:
            features, output = self.extractor(input)

        output=torch.where(output>0.5,output,torch.full_like(output, 0))

        if self.cuda:
            one_hot = torch.sum(output)
            #print(one_hot)
        else:
            one_hot = torch.sum(output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True) # saum : important because we can get_gradients only once the backward step is done.

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  # 1, 2048, 7, 7
        target = features[-1]  # 1,2048,7,7
        target = target.cpu().data.numpy()[0, :]  # 2048, 7, 7

        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 2048
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):  # w:weight,target:feature
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)  # 7,7
        cam = cv2.resize(cam, input.shape[2:])  # 224,224
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

