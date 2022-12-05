'''
num_classes = 1 because using cross-entropy loss.
'''
import torch
import numpy as np
import argparse, json
import os, glob, sys
from time import time
from PIL import Image
from torch.utils.data import DataLoader
from dataloader import Dataset2D, Dataset2D_cityscapes
from unet.unet_model import UNet, GradCam
from utilities import softmax_helper, torch_dice_fn_ce
import cv2

def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['folders'] = [params['common']['img_folder'], params['common']['gt_folder']]
    mydict["checkpoint_restore"] = params['common']['checkpoint_restore']
    mydict["dataset"] = params['common']['dataset']

    if mydict["dataset"] == "ivus":
        mydict['pngimgfolder'] = "/data/saumgupta/ml-cse512/project/ivus-data/test-png/input"
    

    if activity == "train":
        mydict['train_datalist'] = params['train']['train_datalist']
        mydict['validation_datalist'] = params['train']['validation_datalist']
        mydict['output_folder'] = params['train']['output_folder']
        mydict['train_batch_size'] = int(params['train']['train_batch_size'])
        mydict['val_batch_size'] = int(params['train']['val_batch_size'])
        mydict['learning_rate'] = float(params['train']['learning_rate'])
        mydict['num_epochs'] = int(params['train']['num_epochs']) + 1
        mydict['save_every'] = params['train']['save_every']
        mydict["start_epoch"] = params['train']['start_epoch']

    else:
        print("Wrong activity chosen")
        sys.exit()

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    print(activity, mydict)
    return activity, mydict


def set_seed(): # reproductibility 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def show_cam_on_image(torchimg, cammap, step):
    img = torch.squeeze(torchimg).cpu().detach().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * cammap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    img_3ch = np.zeros((img.shape[0],img.shape[1],3))
    for idx in range(3):
        img_3ch[:,:,idx] = img

    cam = heatmap + np.float32(img_3ch)
    cam = cam / np.max(cam)
    cv2.imwrite(os.path.join(mydict['output_folder'], "cam_{}.jpg".format(step)), np.uint8(255 * cam))

    # saum : see bgr/rgb issue? See xiaoling's heatmap code ; heatmap on grayscale code
    # check range of values for heatmap and img --- 1 vs 255


def show_cam_on_image_saum(torchimg, cammap, filename,ch3=False):

    if mydict['dataset'] == 'ivus':
        ip = glob.glob(mydict['pngimgfolder'] + "/" + filename + "*.png")
        assert len(ip) == 1
        ip = ip[0]
        img = cv2.imread(ip, 0) # read grayscale
        #img = torch.squeeze(torchimg).cpu().detach().numpy()
    
    cammap /= np.max(cammap)

    heatmap = cv2.applyColorMap(np.uint8(255 * cammap), cv2.COLORMAP_JET)
    pilimg = Image.fromarray(cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB))
    savename = filename + "_heatmap.png"
    pilimg.save(os.path.join(mydict['output_folder'], savename))

    #img_3ch = np.zeros((img.shape[0],img.shape[1],3))
    #for idx in range(3):
        #img_3ch[:,:,idx] = img

    #super_imposed_img = cv2.addWeighted(heatmap, 0.3, img_3ch.astype(np.uint8), 0.7, 0)
    #super_imposed_img = Image.fromarray(cv2.cvtColor(super_imposed_img,cv2.COLOR_BGR2RGB))

    #savename = filename + "_overlay.png"
    #super_imposed_img.save(os.path.join(mydict['output_folder'], savename))

    # saum : see bgr/rgb issue? See xiaoling's heatmap code ; heatmap on grayscale code
    # check range of values for heatmap and img --- 1 vs 255

'''
def train_func_2d(mydict):
    # Reproducibility, and Cuda setup
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    force_cudnn_initialization()

    # Train Data       
    training_set = Dataset2D(mydict['train_datalist'], mydict['folders'], is_training= True)
    training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],shuffle=True,num_workers=2, drop_last=True)

    # Validation Data
    validation_set = Dataset2D(mydict['validation_datalist'], mydict['folders'], is_training=False)
    validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['val_batch_size'],shuffle=False,num_workers=2, drop_last=False)

    # Network
    network = UNet(n_channels=1, n_classes=mydict['num_classes'], start_filters=64).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=mydict['learning_rate'], weight_decay=0)

    # Load checkpoint (if specified)
    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))

    # Losses
    train_loss_func = torch.nn.BCELoss(size_average = False, reduce=False, reduction=None)
    
    # Train loop
    best_dict = {}
    best_dict['epoch'] = 0
    best_dict['val_loss'] = None
    print("Let the training begin!")
    num_batches = len(training_generator)
    for epoch in range(mydict['num_epochs']):
        torch.cuda.empty_cache() # cleanup
        network.to(device).train() # after .eval() in validation

        avg_train_loss = 0.0
        epoch_start_time = time()

        for step, (patch, mask) in enumerate(training_generator): 
            optimizer.zero_grad()

            patch = patch.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)

            y_pred = torch.sigmoid(network(patch)) # using sigmoid as per xiaoling's implementation
            loss_val = torch.mean(train_loss_func(y_pred, mask))
            avg_train_loss += loss_val

            loss_val.backward()
            optimizer.step()

        avg_train_loss /= num_batches
        epoch_end_time = time()
        print("Epoch {} took {} seconds.\nAverage training loss: {}".format(epoch, epoch_end_time-epoch_start_time, avg_train_loss))

        validation_start_time = time()
        with torch.no_grad():
            network.eval()
            validation_iterator = iter(validation_generator)
            avg_val_loss = 0.0
            for _ in range(len(validation_generator)):
                x, y_gt = next(validation_iterator)
                x = x.to(device, non_blocking=True)
                y_gt = y_gt.to(device, non_blocking=True)

                y_pred = torch.sigmoid(network(x))
                avg_val_loss += torch_dice_fn_ce(y_pred, y_gt)

            avg_val_loss /= len(validation_generator)
        validation_end_time = time()
        print("End of epoch validation took {} seconds.\nAverage validation loss: {}".format(validation_end_time - validation_start_time, avg_val_loss))

        # check for best epoch and save it if it is and print
        if epoch == 0:
            best_dict['epoch'] = epoch
            best_dict['val_loss'] = avg_val_loss
        else:
            if best_dict['val_loss'] < avg_val_loss:
                best_dict['val_loss'] = avg_val_loss
                best_dict['epoch'] = epoch

        if epoch == best_dict['epoch']:
            torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_best.pth"))
        print("Best epoch so far: {}\n".format(best_dict))
        # save checkpoint for save_every
        if epoch % mydict['save_every'] == 0:
            torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + ".pth"))
'''


def cam_func_2d(mydict):
    # Reproducibility, and Cuda setup
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    force_cudnn_initialization()

    # Validation/Test Data
    validation_set = Dataset2D(mydict['validation_datalist'], mydict['folders'], is_training=False)
    validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['val_batch_size'],shuffle=False,num_workers=2, drop_last=False)

    # Network
    network = UNet(n_channels=1, n_classes=mydict['num_classes'], start_filters=64).to(device)

    cam_network = GradCam(model=network, feature_module=network, target_layer_names=["up4"], use_cuda=True)

    # Load checkpoint
    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    else:
        print("Provide checkpoint!")
        sys.exit()

    
    
    # Loop
    torch.cuda.empty_cache() # cleanup
    network.to(device).train() # after .eval() in validation


    for step, (filename, patch, mask) in enumerate(validation_generator): 

        patch = patch.requires_grad_(True).to(device, dtype=torch.float)
        mask = mask.to(device, dtype=torch.float)

        cammask = cam_network(patch, index=None)

        show_cam_on_image_saum(patch, cammask, filename[0].split('.')[0])
        print(step)

def cam_func_2d_cityscapes(mydict):
    # Reproducibility, and Cuda setup
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    force_cudnn_initialization()

    # Validation/Test Data
    validation_set = Dataset2D_cityscapes(mydict['validation_datalist'], mydict['folders'], is_training=False)
    validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['val_batch_size'],shuffle=False,num_workers=4, drop_last=False)

    # Network
    network = UNet(n_channels=3, n_classes=mydict['num_classes'], start_filters=64).to(device)

    cam_network = GradCam(model=network, feature_module=network, target_layer_names=["up4"], use_cuda=True)

    # Load checkpoint
    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    else:
        print("Provide checkpoint!")
        sys.exit()

    
    
    # Loop
    torch.cuda.empty_cache() # cleanup
    network.to(device).train() # after .eval() in validation

    window = 256

    for step, (filename, patch, _) in enumerate(validation_generator): 

        #patch = patch.requires_grad_(True).to(device, dtype=torch.float) # NCHW
        #mask = mask.to(device, dtype=torch.float)

        np_patch = patch.cpu().detach().numpy()
        print(np_patch.shape)
        for xs in range(0,np_patch.shape[2],window):
            for ys in range(0,np_patch.shape[3],window):
                minipatch_np = np_patch[:,:,xs:xs+window,ys:ys+window]
                print(minipatch_np.shape)
                minipatch_torch = torch.from_numpy(minipatch_np).requires_grad_(True).to(device, dtype=torch.float)
                cammask = cam_network(minipatch_torch, index=None)

                show_cam_on_image_saum(minipatch_torch, cammask, filename[0]+"_{}_{}".format(xs,ys), ch3=True)
                print(xs,ys)
        print(step)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    if activity == "train":
        if mydict["dataset"] == "city":
            cam_func_2d_cityscapes(mydict)
        elif mydict["dataset"] == "ivus":
            cam_func_2d(mydict)