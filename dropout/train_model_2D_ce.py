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
from unet.unet_model import UNet
from utilities import softmax_helper, torch_dice_fn_ce


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

    elif activity == "test":
        mydict['test_datalist'] = params['test']['test_datalist']
        mydict['output_folder'] = params['test']['output_folder']

    else:
        print("Wrong activity chosen")
        sys.exit()

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
    best_dict['epoch'] = None
    best_dict['val_loss'] = None
    print("Let the training begin!")
    num_batches = len(training_generator)
    for epoch in range(mydict["start_epoch"], mydict['num_epochs']):
        torch.cuda.empty_cache() # cleanup
        network.to(device).train() # after .eval() in validation

        avg_train_loss = 0.0
        epoch_start_time = time()

        for step, (_,patch, mask) in enumerate(training_generator): 
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
                _, x, y_gt = next(validation_iterator)
                x = x.to(device, non_blocking=True)
                y_gt = y_gt.to(device, non_blocking=True)

                y_pred = torch.sigmoid(network(x))
                avg_val_loss += torch_dice_fn_ce(y_pred, y_gt)

            avg_val_loss /= len(validation_generator)
        validation_end_time = time()
        print("End of epoch validation took {} seconds.\nAverage validation loss: {}".format(validation_end_time - validation_start_time, avg_val_loss))

        # check for best epoch and save it if it is and print
        if epoch == mydict["start_epoch"]:
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


def train_func_2d_cityscapes(mydict):
    # Reproducibility, and Cuda setup
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    force_cudnn_initialization()

    # Train Data       
    training_set = Dataset2D_cityscapes(mydict['train_datalist'], mydict['folders'], is_training= True)
    training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],shuffle=True,num_workers=4, drop_last=True)

    # Validation Data
    validation_set = Dataset2D_cityscapes(mydict['validation_datalist'], [mydict['folders'][0].replace("train", "val"), mydict['folders'][1].replace("train", "val")], is_training=False)
    validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['val_batch_size'],shuffle=False,num_workers=4, drop_last=False)

    # Network
    network = UNet(n_channels=3, n_classes=mydict['num_classes'], start_filters=64).to(device)

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
    best_dict['epoch'] = None
    best_dict['val_loss'] = None
    print("Let the training begin!")
    num_batches = len(training_generator)
    for epoch in range(mydict["start_epoch"], mydict['num_epochs']):
        torch.cuda.empty_cache() # cleanup
        network.to(device).train() # after .eval() in validation

        avg_train_loss = 0.0
        epoch_start_time = time()

        for step, (_,patch, mask) in enumerate(training_generator): 
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
                _,x, y_gt = next(validation_iterator)
                x = x.to(device, non_blocking=True)
                y_gt = y_gt.to(device, non_blocking=True)

                y_pred = torch.sigmoid(network(x))
                avg_val_loss += torch_dice_fn_ce(y_pred, y_gt)

            avg_val_loss /= len(validation_generator)
        validation_end_time = time()
        print("End of epoch validation took {} seconds.\nAverage validation loss: {}".format(validation_end_time - validation_start_time, avg_val_loss))

        # check for best epoch and save it if it is and print
        if epoch == mydict["start_epoch"]:
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


def SaveFiles(output_folder, filename, np_vol):
    savename = os.path.join(output_folder, filename+".png")
    print("Saving {}".format(savename))

    pil_img = Image.fromarray(np_vol.astype(np.uint8))
    pil_img.save(savename)



def test_func(mydict):
    print("Inference!")
    #device = torch.device("cuda")
    #print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    if mydict['dataset'] == "ivus":
        network = UNet(n_channels=1, n_classes=mydict['num_classes'], start_filters=64)
        test_set = Dataset2D(mydict['test_datalist'], mydict['folders'], is_training=False)

    elif mydict['dataset'] == "city":
        network = UNet(n_channels=3, n_classes=mydict['num_classes'], start_filters=64)
        test_set = Dataset2D_cityscapes(mydict['test_datalist'], mydict['folders'], is_training=False)

    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
    else:
        print("Checkpoint not specified! Aborting...")
        sys.exit()

    network = network.eval()
    num_samples = test_set.__len__()
    print("Number of volumes: {}".format(num_samples))
    for ind in range(num_samples):
        filename, torch_img, _ = test_set.__getitem__(ind)

        torch_img = torch.unsqueeze(torch_img, dim=0) # to include batch=1 dimension
    
        y_pred = torch.sigmoid(network(torch_img)) # NCDHW

        y_pred = torch.squeeze(y_pred).detach().numpy()
        
        #y_pred[y_pred < 0.5] = 0.
        #y_pred[y_pred >= 0.5] = 255.
        y_pred = np.clip(y_pred * 255., 0., 255.)

        SaveFiles(mydict['output_folder'], filename+"_3", y_pred)





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
            train_func_2d_cityscapes(mydict)
        elif mydict["dataset"] == "ivus":
            train_func_2d(mydict)

    elif activity == "test":
        test_func(mydict)