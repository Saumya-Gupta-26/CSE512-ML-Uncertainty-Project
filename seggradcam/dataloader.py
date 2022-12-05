# load all into cpu
# do cropping to patch size
# restrict number of slices (see null slices)
# normalize range
# test code by outputting few patches
# training, testing, val
import torch
import SimpleITK as sitk
from PIL import Image
import os, glob, sys
import numpy as np

class Dataset2D(torch.utils.data.Dataset):
    def __init__(self, listpath, folderpaths, is_training):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]

        self.patchsize = 128
        self.is_training = is_training

        self.dataCPU = {}
        self.dataCPU['filename'] = []
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.indices = [] # contains tuple i, j ---> i is volume number, j is slice number

        self.loadCPU()
        print("Length of dataset (num of 2D slices): {}".format(len(self.indices)))

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist):
            minslice = None
            maxslice = None

            components = entry.split(',')
            filename = components[0]
            if len(components) > 1:
                minslice = int(components[1])
            if len(components) > 2:
                maxslice = int(components[2])

            arrayimage = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.imgfolder, filename+"_0000.nii.gz"))).astype(np.float32)
            arrayimage_gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.gtfolder, filename+".nii.gz"))).astype(np.float32)
            arrayimage_gt[arrayimage_gt > 0.] = 1.

            assert arrayimage.shape == arrayimage_gt.shape #DHW

            if minslice is None:
                minslice = 0
            if maxslice is None:
                maxslice = arrayimage.shape[0]

            arrayimage = arrayimage[minslice:maxslice]
            arrayimage_gt = arrayimage_gt[minslice:maxslice]

            assert arrayimage.shape == arrayimage_gt.shape

            #normalize within a slice
            for j in range(arrayimage.shape[0]):
                meanval = arrayimage[j].mean()
                stdval = arrayimage[j].std()

                if stdval == 0.0:
                    arrayimage[j] = arrayimage[j]/meanval
                else:
                    arrayimage[j] = (arrayimage[j] - meanval) / stdval

                self.indices.append((i,j))

            #cpu store
            self.dataCPU['filename'].append(filename)
            self.dataCPU['image'].append(arrayimage)
            self.dataCPU['label'].append(arrayimage_gt)
            print("Filename: {}, minslice: {}, maxslice: {}, modshape: {}\ni: {}, len of dataCPU so far: {}\nmin-intensity: {}, max-intensity: {}\n\n".format(filename, minslice, maxslice, arrayimage_gt.shape,i, len(self.dataCPU['image']), np.min(arrayimage), np.max(arrayimage)))


    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        ivolume, jslice = self.indices[index]

        np_img = self.dataCPU['image'][ivolume][jslice] #HW
        np_gt = self.dataCPU['label'][ivolume][jslice] #HW

        if self.is_training:
            H, W = np_img.shape
            corner_h = np.random.randint(low=0, high=H-self.patchsize)
            corner_w = np.random.randint(low=0, high=W-self.patchsize)

            np_img = np_img[corner_h:corner_h+self.patchsize, corner_w:corner_w+self.patchsize]
            np_gt = np_gt[corner_h:corner_h+self.patchsize, corner_w:corner_w+self.patchsize]

        torch_img = torch.unsqueeze(torch.from_numpy(np_img),dim=0) # CHW
        torch_gt = torch.unsqueeze(torch.from_numpy(np_gt),dim=0) # CHW

        return self.dataCPU['filename'][ivolume], torch_img, torch_gt 

#skip patch which doesn't have human
#skip image which doesnt have human --- 2500 out of 2975 images have human/rider
# need to check for which cases 128 min size isn't there
class Dataset2D_cityscapes(torch.utils.data.Dataset):
    def __init__(self, listpath, folderpaths, is_training):

        self.is_training = is_training

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]

        self.img_suffix = "_leftImg8bit.png"
        self.gt_suffix = "_gtFine_labelIds_saum.png"

        self.patchsize = 128


        self.dataCPU = {}
        self.dataCPU['imgpath'] = []
        self.dataCPU['gtpath'] = []
        self.dataCPU['coords'] = []

        self.indices = [] # contains tuple i, j ---> i is volume number, j is slice number

        self.loadCPU() # only stores filepaths
        print("Length of dataset (num of 2D slices): {}".format(len(self.indices)))


    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        cnt = 0

        for i, entry in enumerate(mylist):

            components = entry.split(',')

            if self.is_training:
                _, tl_x, tl_y, br_x, br_y = components
                tl_x = int(tl_x)
                tl_y = int(tl_y)
                br_x = int(br_x)
                br_y = int(br_y)

                xsub = br_x - tl_x
                ysub = br_y - tl_y

                if (xsub < 0) or (ysub < 0):
                    continue

            filename = components[0]

            imgpath = glob.glob(self.imgfolder + '/*/' + filename + self.img_suffix)
            gtpath = glob.glob(self.gtfolder + '/*/' + filename + self.gt_suffix)

            assert len(imgpath) == 1
            assert len(gtpath) == 1

            imgpath = imgpath[0]
            gtpath = gtpath[0]

            if self.is_training:
                img_shape = np.asarray(Image.open(gtpath)).shape

                if (xsub <= self.patchsize):
                    tl_x = max(0, tl_x - self.patchsize)
                    br_x = min(br_x + self.patchsize, img_shape[0])

                if (ysub <= self.patchsize):
                    tl_y = max(0, tl_y - self.patchsize)
                    br_y = min(br_y + self.patchsize, img_shape[1])

                self.dataCPU['coords'].append([tl_x, tl_y, br_x, br_y])


            #cpu store
            self.dataCPU['imgpath'].append(imgpath)
            self.dataCPU['gtpath'].append(gtpath)
            
            print("Filename: {}\n".format(filename))
            self.indices.append((cnt,0))
            cnt += 1 # because of 'continue' in the code, 'i' is not equal to 'cnt'

        assert len(self.dataCPU['imgpath']) == len(self.dataCPU['gtpath'])
        if self.is_training:
            assert len(self.dataCPU['imgpath']) == len(self.dataCPU['coords'])


    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        ivolume, _ = self.indices[index]

        imgpath = self.dataCPU['imgpath'][ivolume]
        gtpath = self.dataCPU['gtpath'][ivolume]

        np_img = np.asarray(Image.open(imgpath)).astype(np.float32)
        np_gt = np.asarray(Image.open(gtpath)).astype(np.float32) / 255.

        assert np_img.shape[0] == np_gt.shape[0]
        assert np_img.shape[1] == np_gt.shape[1]

        if self.is_training:

            H, W, _ = np_img.shape

            coords = self.dataCPU['coords'][ivolume]
            corner_h = np.random.randint(low=coords[0], high=coords[2]-self.patchsize)
            corner_w = np.random.randint(low=coords[1], high=coords[3]-self.patchsize)

            np_img = np_img[corner_h:corner_h+self.patchsize, corner_w:corner_w+self.patchsize, :]
            np_gt = np_gt[corner_h:corner_h+self.patchsize, corner_w:corner_w+self.patchsize]

            assert np_img.shape[0] == np_gt.shape[0]
            assert np_img.shape[1] == np_gt.shape[1]

        np_img = np.transpose(np_img, (2, 0, 1))
        torch_img = torch.from_numpy(np_img) # CHW
        torch_gt = torch.unsqueeze(torch.from_numpy(np_gt),dim=0) # CHW

        return imgpath.split('/')[-1].replace(self.img_suffix, ""), torch_img, torch_gt 





# uses filelist for input # make sure to add the .requires_grad_(True)
class Dataset2D_Inference():
    def __init__(self, listpath, folderpaths):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]

        self.dataCPU = {}
        self.dataCPU['filename'] = []
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]


        for i, entry in enumerate(mylist):

            components = entry.split(',')
            filename = components[0]

            arrayimage = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.imgfolder, filename+"_0000.nii.gz"))).astype(np.float32)
            sitkimage_gt = sitk.ReadImage(os.path.join(self.gtfolder, filename+".nii.gz"))

            #normalize within a slice
            for j in range(arrayimage.shape[0]):
                meanval = arrayimage[j].mean()
                stdval = arrayimage[j].std()

                if stdval == 0.0:
                    arrayimage[j] = arrayimage[j]/meanval
                else:
                    arrayimage[j] = (arrayimage[j] - meanval) / stdval


            #cpu store
            filename = filename+".nii.gz"
            self.dataCPU['filename'].append(filename)
            self.dataCPU['image'].append(arrayimage)
            self.dataCPU['label'].append(sitkimage_gt)
            print("Filename: {}\nmin-intensity: {}, max-intensity: {}\n\n".format(filename, np.min(arrayimage), np.max(arrayimage)))


    def __len__(self): # total number of 3D Volumes
        return len(self.dataCPU['filename'])

    def __getitem__(self, index): # return np_img, sitk_gt
        
        filename = self.dataCPU['filename'][index]
        np_img = self.dataCPU['image'][index] #DHW
        sitk_gt = self.dataCPU['label'][index]

        return filename, np_img, sitk_gt 




# uses folder for input
class Dataset2D_Inference_2():
    def __init__(self, folderpaths):

        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]

        self.dataCPU = {}
        self.dataCPU['filename'] = []
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.loadCPU()

    def loadCPU(self):
        mylist = os.listdir(self.gtfolder)
        mylist = [ml for ml in mylist if 'nii.gz' in ml]

        for i, entry in enumerate(mylist):
            filename = entry

            arrayimage = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.imgfolder, filename.replace(".nii.gz", "_0000.nii.gz")))).astype(np.float32)
            sitkimage_gt = sitk.ReadImage(os.path.join(self.gtfolder, filename))

            #normalize within a slice
            for j in range(arrayimage.shape[0]):
                meanval = arrayimage[j].mean()
                stdval = arrayimage[j].std()

                if stdval == 0.0:
                    arrayimage[j] = arrayimage[j]/meanval
                else:
                    arrayimage[j] = (arrayimage[j] - meanval) / stdval


            #cpu store
            self.dataCPU['filename'].append(filename)
            self.dataCPU['image'].append(arrayimage)
            self.dataCPU['label'].append(sitkimage_gt)
            print("Filename: {}\nmin-intensity: {}, max-intensity: {}\n\n".format(filename, np.min(arrayimage), np.max(arrayimage)))


    def __len__(self): # total number of 3D Volumes
        return len(self.dataCPU['filename'])

    def __getitem__(self, index): # return np_img, sitk_gt
        
        filename = self.dataCPU['filename'][index]
        np_img = self.dataCPU['image'][index] #DHW
        sitk_gt = self.dataCPU['label'][index]

        return filename, np_img, sitk_gt 




if __name__ == "__main__":
    flag = "training"

    if flag == "training":
        training_set = Dataset2D_3slice('data-lists/fold0/train-list.csv', ['/home/saumya/aorta-segmentation/baseline/nnunet/data/data-format-for-all/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/imagesTr', '/home/saumya/aorta-segmentation/baseline/nnunet/data/data-format-for-all/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/labelsTr'])

        training_generator = torch.utils.data.DataLoader(training_set,batch_size=1,shuffle=True,num_workers=1)

        # one epoch
        for local_batch, local_gt in training_generator:

            # local_batch, local_gt = local_batch.to(device), local_labels.to(device)
            pass
    
    elif flag == "validation":
        validation_set = Dataset2D_3slice('data-lists/fold0/validation-list.csv', ['/home/saumya/aorta-segmentation/baseline/nnunet/data/data-format-for-all/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/imagesTr', '/home/saumya/aorta-segmentation/baseline/nnunet/data/data-format-for-all/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/labelsTr'])

        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=12,shuffle=True,num_workers=6)

        # one epoch
        for local_batch, local_gt in validation_generator:
            # local_batch, local_gt = local_batch.to(device), local_labels.to(device)
            pass