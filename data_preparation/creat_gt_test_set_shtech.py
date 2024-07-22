import numpy as np
from scipy.ndimage import gaussian_filter
import os
import cv2
import scipy.io as sio
import numpy as np


def get_density_map_gaussian(im, points):
    im_density = np.zeros_like(im, dtype=np.float32)
    h, w = im_density.shape[:2]

    if len(points) == 0:
        return im_density

    if points.ndim == 1:  # Assuming points is a 1D numpy array
        num_points = len(points) // 2
        x_coords = points[:num_points]
        y_coords = points[num_points:]

        for k in range(num_points):
            x = int(np.clip(np.floor(x_coords[k]).astype(np.int32), 0, w - 1))
            y = int(np.clip(np.floor(y_coords[k]).astype(np.int32), 0, h - 1))
            im_density[y, x] += 1  # Example: Increment density at point (y, x)

        return im_density

    # Handle other cases as needed

    return im_density


dataset = 'B'
dataset_name = f'shanghaitech_part_{dataset}'
path = f'../data/original/shanghaitech/part_{dataset}_final/test_data/images/'
gt_path = f'../data/original/shanghaitech/part_{dataset}_final/test_data/ground_truth/'
gt_path_csv = f'../data/original/shanghaitech/part_{dataset}_final/test_data/ground_truth_csv/'

os.makedirs(gt_path_csv, exist_ok=True)

num_images = 182 if dataset == 'A' else 316

for i in range(1, num_images + 1):
    if i % 10 == 0:
        print(f'Processing {i}/{num_images} files')

    mat = sio.loadmat(f'{gt_path}GT_IMG_{i}.mat')
    image_info = mat['image_info']
    annPoints = image_info[0][0][0][0][0][0]

    input_img_name = f'{path}IMG_{i}.jpg'
    im = cv2.imread(input_img_name)
    h, w, c = im.shape

    if c == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im_density = get_density_map_gaussian(im, annPoints)

    csv_file = f'{gt_path_csv}IMG_{i}.csv'
    np.savetxt(csv_file, im_density, delimiter=',')
