import numpy as np
import os
import cv2
import scipy.io as sio

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

seed = 95461354
np.random.seed(seed)

N = 9
dataset = 'B'
dataset_name = f'shanghaitech_part_{dataset}_patches_{N}'
path = f'../data/original/shanghaitech/part_{dataset}_final/train_data/images/'
output_path = '../data/formatted_trainval/'
train_path_img = os.path.join(output_path, f'{dataset_name}/train/')
train_path_den = os.path.join(output_path, f'{dataset_name}/train_den/')
val_path_img = os.path.join(output_path, f'{dataset_name}/val/')
val_path_den = os.path.join(output_path, f'{dataset_name}/val_den/')
gt_path = f'../data/original/shanghaitech/part_{dataset}_final/train_data/ground_truth/'

os.makedirs(output_path, exist_ok=True)
os.makedirs(train_path_img, exist_ok=True)
os.makedirs(train_path_den, exist_ok=True)
os.makedirs(val_path_img, exist_ok=True)
os.makedirs(val_path_den, exist_ok=True)

num_images = 300 if dataset == 'A' else 400
num_val = int(np.ceil(num_images * 0.1))
indices = np.random.permutation(num_images) + 1

for idx, i in enumerate(indices):
    if (idx + 1) % 10 == 0:
        print(f'Processing {idx + 1}/{num_images} files')

    # Load the ground truth points
    mat = sio.loadmat(f'{gt_path}GT_IMG_{i}.mat')
    image_info = mat['image_info']
    annPoints = image_info[0][0][0][0][0][0]

    # Ensure annPoints is reshaped if needed (for safety, though it should be 2D)
    annPoints = annPoints.reshape(-1, 2)

    input_img_name = f'{path}IMG_{i}.jpg'
    im = cv2.imread(input_img_name)
    h, w, c = im.shape

    if c == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    wn2 = w // 8
    hn2 = h // 8
    wn2 = 8 * (wn2 // 8)
    hn2 = 8 * (hn2 // 8)

    if w <= 2 * wn2:
        im = cv2.resize(im, (2 * wn2 + 1, h))
        annPoints[:, 0] = annPoints[:, 0] * (2 * wn2 / w)

    if h <= 2 * hn2:
        im = cv2.resize(im, (w, 2 * hn2 + 1))
        annPoints[:, 1] = annPoints[:, 1] * (2 * hn2 / h)

    h, w = im.shape[:2]
    a_w = wn2 + 1
    b_w = w - wn2
    a_h = hn2 + 1
    b_h = h - hn2

    im_density = get_density_map_gaussian(im, annPoints)

    for j in range(N):
        x = np.random.randint(a_w, b_w)
        y = np.random.randint(a_h, b_h)
        x1 = x - wn2
        y1 = y - hn2
        x2 = x + wn2
        y2 = y + hn2

        im_sampled = im[y1:y2 + 1, x1:x2 + 1]
        im_density_sampled = im_density[y1:y2 + 1, x1:x2 + 1]

        # Filter annPoints for points inside the sampled patch
        annPoints_sampled = annPoints[(annPoints[:, 0] > x1) & (annPoints[:, 0] < x2) &
                                      (annPoints[:, 1] > y1) & (annPoints[:, 1] < y2)]
        annPoints_sampled[:, 0] -= x1
        annPoints_sampled[:, 1] -= y1

        img_idx = f'{i}_{j + 1}'

        if idx < num_val:
            cv2.imwrite(f'{val_path_img}{img_idx}.jpg', im_sampled)
            np.savetxt(f'{val_path_den}{img_idx}.csv', im_density_sampled, delimiter=',')
        else:
            cv2.imwrite(f'{train_path_img}{img_idx}.jpg', im_sampled)
            np.savetxt(f'{train_path_den}{img_idx}.csv', im_density_sampled, delimiter=',')
