import os
import os.path

import torch.utils.data as data
from PIL import Image
import random
import torch
import numpy as np
import pdb 

from glob import glob

from torchvision.transforms import v2 as transforms


def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

# return image triple pairs in video and return single image
class CrossPairwiseImg(data.Dataset):
    def __init__(self, root, is_train=True, random_flip = False, enable_color_aug=False, height=416, width=416, crop_h=416, crop_w=416):
        self.img_root, self.video_root = self.split_root(root)



        self.is_train = is_train
        self.enable_color_aug = enable_color_aug
        self.random_flip = random_flip

        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_ext = '.jpg'
        self.label_ext = '.png'

        self.full_res_shape = None

        self.height = height
        self.width = width
        if crop_h is None or crop_w is None or not self.is_train:
            self.crop_h = self.height
            self.crop_w = self.width
        else:
            self.crop_h = crop_h
            self.crop_w = crop_w
     

        self.frame_idxs = [0, 1, -1]
        self.num_scales = 4

        self.fx = 1534.56289
        self.fy = 1534.56289
        self.u0 = 960
        self.v0 = 540
        
        self.to_tensor = transforms.ToTensor()
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.crop_h // s, self.crop_w // s))

        self.resize_full = transforms.Resize((self.height, self.width))

        print(self.video_root)

        self.num_video_frame = 0
        # get all frames from video datasets
        self.videoImg_list = self.generateImgFromVideo(self.video_root)
        print('Total video frames is {}.'.format(self.num_video_frame))
        # get all frames from image datasets
        if len(self.img_root) > 0:
            self.singleImg_list = self.generateImgFromSingle(self.img_root)
            print('Total single image frames is {}.'.format(len(self.singleImg_list)))
        
        print(f"{self.video_root[0][0]} has {len(os.listdir(self.video_root[0][0]))} videos.")
        
        
    def get_ref(self, img_path):
        video = img_path.split('/')[-3]
        img_name = img_path.split('/')[-1]

        path = os.path.join(self.reflection_root, video, img_name)
        try:
            ref = Image.open(path).convert('RGB')
        except:
            # ref = Image.open(path.replace('jpg', 'png')).convert('RGB')
            ref = Image.open(path.replace('.jpg', '_fake_Rs_03.png')).convert('RGB')

        return ref 


    def __getitem__(self, index):
        manual_random = random.random()  # random for transformation
        # pair in video
        exemplar_path, exemplar_gt_path, videoStartIndex, videoLength = self.videoImg_list[index]  # exemplar
        # sample from same video
        query_index = np.random.randint(videoStartIndex, videoStartIndex + videoLength)
        if query_index == index:
            query_index = np.random.randint(videoStartIndex, videoStartIndex + videoLength)

        # # sample from different video
        # while True:
        #     other_index = np.random.randint(0, self.__len__())
        #     if other_index < videoStartIndex or other_index > videoStartIndex + videoLength - 1:
        #         break  # find image from different video

        # print(index, query_index)
        
        other_index = index + 1
        if other_index >= videoStartIndex + videoLength - 1: # index is the last frame in video
            other_index = videoStartIndex # select the first frame in video
        ## swap
        query_index, other_index = other_index, query_index
        
        query_path, query_gt_path, videoStartIndex2, videoLength2 = self.videoImg_list[query_index]  # query
        if videoStartIndex != videoStartIndex2 or videoLength != videoLength2:
            raise TypeError('Something wrong')
        other_path, other_gt_path, videoStartIndex3, videoLength3 = self.videoImg_list[other_index]  # other
        # if videoStartIndex != videoStartIndex3 or videoLength != videoLength3:
        #     raise TypeError('Something wrong')
        # if videoStartIndex == videoStartIndex3:
        #     raise TypeError('Something wrong')
        # single image in image dataset
        if len(self.img_root) > 0:
            single_idx = np.random.randint(0, videoLength)
            single_image_path, single_gt_path = self.singleImg_list[single_idx]  # single image
        
        # read image and gt
        exemplar = Image.open(exemplar_path).convert('RGB')
        query = Image.open(query_path).convert('RGB')
        other = Image.open(other_path).convert('RGB')
        exemplar_gt = Image.open(exemplar_gt_path).convert('L')
        query_gt = Image.open(query_gt_path).convert('L')
        other_gt = Image.open(other_gt_path).convert('L')
        if len(self.img_root) > 0:
            single_image = Image.open(single_image_path).convert('RGB')
            single_gt = Image.open(single_gt_path).convert('L')

        self.full_res_shape = exemplar.size
        self.u0 = self.full_res_shape[0] / 2
        self.v0 = self.full_res_shape[1] / 2

        do_color_aug = self.is_train and random.random() < 0.5 and self.enable_color_aug
        do_flip = self.is_train and random.random() < 0.5 and self.random_flip

        if do_flip:
            exemplar = exemplar.transpose(Image.FLIP_LEFT_RIGHT)
            query = query.transpose(Image.FLIP_LEFT_RIGHT)
            other = other.transpose(Image.FLIP_LEFT_RIGHT)
            exemplar_gt = exemplar_gt.transpose(Image.FLIP_LEFT_RIGHT)
            query_gt = query_gt.transpose(Image.FLIP_LEFT_RIGHT)
            other_gt = other_gt.transpose(Image.FLIP_LEFT_RIGHT)

        sample = {
            ("color", 0, -1): exemplar,
            ("color", 1, -1): query,
            ("color", -1, -1): other,
            ("gt", 0): exemplar_gt,
            ("gt", 1): query_gt,
            ("gt", -1): other_gt
        }


        if self.is_train:
            sample = self.random_crop(sample, do_flip)

        self.preprocess(sample, do_color_aug)
            

        sample['exemplar_path'] = exemplar_path
        sample['query_path'] = query_path
        sample['other_path'] = other_path 

        for i in self.frame_idxs:
            del sample[("color", i, -1)]
            del sample[("color_full", i, -1)]
            del sample[("gt_full", i)]


        return sample

    def generateImgFromVideo(self, root):
        imgs = []
        root = root[0]  # assume that only one video dataset
        video_list = listdirs_only(os.path.join(root[0]))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root[0], video, self.input_folder)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            for img in img_list:
                # videoImgGt: (img, gt, video start index, video length)
                videoImgGt = (os.path.join(root[0],video,  self.input_folder, img + self.img_ext),
                        os.path.join(root[0], video, self.label_folder, img + self.label_ext), self.num_video_frame, len(img_list))
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)

        return imgs

    def generateImgFromSingle(self, root):
        imgs = []
        for sub_root in root:
            tmp = self.generateImagePair(sub_root[0])
            imgs.extend(tmp)  # deal with image case
            print('Image number of ImageSet {} is {}.'.format(sub_root[2], len(tmp)))

        return imgs

    def generateImagePair(self, root):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, self.input_folder)) if f.endswith(self.img_ext)]
        if len(img_list) == 0:
            raise IOError('make sure the dataset path is correct')
        return [(os.path.join(root, self.input_folder, img_name + self.img_ext), os.path.join(root, self.label_folder, img_name + self.label_ext))
            for img_name in img_list]

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        video_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                img_root_list.append(tmp)
            elif tmp[1] == 'video':
                video_root_list.append(tmp)
            else:
                raise TypeError('you should input video or image')
        return img_root_list, video_root_list

    def __len__(self):
        return len(self.videoImg_list)//2*2
    
    def random_crop(self, inputs, do_flip):
        w, h = inputs[("color", 0, -1)].size
        th, tw = self.crop_h, self.crop_w

        # assert h <= w and th <= tw
        if w < tw or h < th:

            raise NotImplementedError


        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        crop_region = (x1, y1, x1 + tw, y1 + th)

        for i in self.frame_idxs:
            img = inputs[("color", i, -1)]
            inputs[("color_full", i, -1)] = img
            inputs[("gt_full", i)] = inputs[("gt", i)]
            if w != tw or h != th:
                inputs[("color", i, -1)] = img.crop(crop_region)
                inputs[("gt", i)] = inputs[("gt", i)].crop(crop_region)
                

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.get_K(x1, y1, do_flip)

            K[0, :] /= (2 ** scale)
            K[1, :] /= (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        return inputs

    def preprocess(self, inputs, do_color_aug = False):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
        else:
            color_aug = lambda x: x

        for k in list(inputs):
            if len(k) == 3:
                n, im, i = k
                if n == "color":
                    for i in range(self.num_scales):
                        inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                if n == "color_full":
                    inputs[(n, im, 0)] = self.resize_full(inputs[(n, im, -1)])
            elif len(k) == 2:
                n, im = k
                if n == "gt_full":
                    inputs[("gt_resized", im)] = self.resize_full(inputs[(n, im)])


        for k in list(inputs):
            f = inputs[k]
            if len(k) == 3:
                n, im, i = k
                if "color" in n:
                    inputs[(n, im, i)] = self.to_tensor(f)
                    if i == 0:
                        processed_f = normalize(self.to_tensor(color_aug(f)))
                        inputs[(n + "_aug", im, i)] = processed_f
            elif len(k) == 2:
                n, im = k
                if "gt" in n:
                    inputs[(n, im)] = self.to_tensor(f)
            
    

    def get_K(self, u_offset, v_offset, do_flip):
        u0 = self.u0
        v0 = self.v0
        if do_flip:
            u0 = self.full_res_shape[0] - u0
            v0 = self.full_res_shape[1] - v0

        return np.array([[self.fx, 0, u0 - u_offset, 0],
                         [0, self.fy, v0 - v_offset, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)

