import logging
import numpy as np
import torch
import os

from PIL import Image
import csv
from scipy.spatial.transform import Rotation

    
def search_image(images_dir, image_name):
    image_path = os.path.join(images_dir, image_name + ".png")
    if os.path.exists(image_path):
        return image_path
    image_path = os.path.join(images_dir, image_name + ".jpg")
    if os.path.exists(image_path):
        return image_path
    image_path = os.path.join(images_dir, image_name + ".JPG")
    if os.path.exists(image_path):
        return image_path
    return None

class DatasetPhoto(torch.utils.data.Dataset):

    def __init__(self, images_dir: str, csv_path: str, mask_dir="", depth_dir="", preload=False):
        # self.image_list = []
        self.image_paths = []
        self.mask_paths = []
        self.depth_paths = []
        self.img_names = []
        self.camera_focal35mm = []
        self.camera_positions = []
        self.camera_rotations = []
        self.preload = preload
        self.image_tensors = []
        self.image_list = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                image_name_with_ext = row[0]
                image_name = row[0].split(".")[0]
                image_path = search_image(images_dir, image_name)
                if image_path is None:
                    continue
                self.image_paths.append(image_path)
                if mask_dir != "":
                    mask_path = os.path.join(mask_dir, image_name_with_ext + ".mask.png")
                    self.mask_paths.append(mask_path)
                if depth_dir != "":
                    depth_path = os.path.join(depth_dir, image_name_with_ext + ".depth.exr")
                    self.depth_paths.append(depth_path)
                self.img_names.append(image_name)

                x, y, alt, heading, pitch, roll, f, px, py, k1, k2, k3, k4, t1, t2 = map(float, row[1:])
                self.camera_focal35mm.append(f)

                pos = torch.tensor([[x, y, alt]], dtype=torch.float32, device="cpu")
                self.camera_positions.append(pos)

                roll *= -1
                pitch *= -1
                quat = Rotation.from_euler('YXZ', [roll, pitch, heading], degrees=True).as_quat()   # x,y,z,w format
                quat = quat[[3, 0, 1, 2]]   # w,x,y,z format
                quat = torch.tensor(quat, dtype=torch.float32, device="cpu")
                self.camera_rotations.append(quat)

        if self.preload:
            for image_path in self.image_paths:
                pil_img = Image.open(image_path)
                self.image_list.append(pil_img)
                image_tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
                image_tensor.share_memory_()
                self.image_tensors.append(image_tensor)

    def __len__(self):
        return len(self.image_paths)
        # return 16

    def __getitem__(self, itr):
        image_path = self.image_paths[itr]
        if self.preload:
            pil_img = self.image_list[itr]
            image_tensor = self.image_tensors[itr]
        else:
            pil_img = Image.open(image_path)
            image_tensor = torch.from_numpy(np.array(pil_img))
            # 由于reality capture在undistort之后图像有黑边，所以需要mask掉，jpg格式有损，所以用大于6来判断
            undistort_mask = torch.sum(image_tensor, dim=-1, keepdim=True) > 0
            image_tensor = image_tensor.float() / 255.0
            mask = None
            if len(self.mask_paths) > 0:
                mask_img = Image.open(self.mask_paths[itr])
                mask = torch.from_numpy(np.array(mask_img))[..., None]
                image_tensor *= mask
        
        return {
            'image': pil_img,
            "image_tensor": image_tensor,
            'image_name': self.img_names[itr],
            "focal35mm": self.camera_focal35mm[itr],
            "camera_position": self.camera_positions[itr],
            "camera_rotation": self.camera_rotations[itr],
            "image_index": itr,
            "mask": mask.float(),
            "undistort_mask": undistort_mask.float()
        }
    
    @staticmethod
    def collate_fn(batch):
        return batch

