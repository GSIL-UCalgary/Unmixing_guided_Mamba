import torch
import numpy as np
from torchvision import transforms
import pdb
import matplotlib.pyplot as plt

def percentile_stretch(band, lower=2, upper=98):
    """Apply percentile stretch to a single band"""
    p_low, p_high = np.percentile(band, (lower, upper))
    
    # 防止除0
    if p_high - p_low < 1e-8:
        return np.zeros_like(band)
    
    band = (band - p_low) / (p_high - p_low)
    band = np.clip(band, 0, 1)
    return band



def ImageStretching(image):
    channels = image.shape[2]
    band_list = []
    image_data = (image - np.min(image)) / (np.max(image) - np.min(image))
    H, W, C = image_data.shape
    data = image_data.reshape(-1, C)
    from sklearn.decomposition import PCA

    # data shape: [N_pixels, 200]
    pca = PCA(n_components=3, whiten=False)

    data_pca = pca.fit_transform(data)   # [N_pixels, 30]
    image_pca = data_pca.reshape(H, W, 3)
    for i in range(3):
        band = image_pca[:, :, i]
        image_pca[:, :, i] = percentile_stretch(band, 2, 98)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(image_pca)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("PCA.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    # for i in range(channels):
    #     band_data = image[:,:,i]
    #     band_min = np.percentile(band_data,2)
    #     band_max = np.percentile(band_data,98)
    #     band_data = (band_data - band_min) / (band_max - band_min)
    #     # plt.imshow(band_data)
    #     # plt.show()
    #     band_list.append(band_data)
    # image_data = np.stack(band_list, axis=-1)
    # image_data = np.clip(image_data, 0, 1)
    # image_data = (image_data * 255).astype(np.uint8)
    # image_data = np.uint8(image_data)

    red_idx = 70
    green_idx = 60
    blue_idx = 40

    rgb = image_data[:, :, [red_idx, green_idx, blue_idx]].astype(float)

    for i in range(3):
        band = rgb[:,:,i]
        rgb[:,:,i] = (band - band.min()) / (band.max() - band.min())

    plt.imshow(rgb)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("RGB.pdf",  dpi=300, bbox_inches='tight')
    return image_data


def normlize3D(image,use_group=False,group_num=4):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    size = image.shape
    if size[2]!=3:
        image_norms = []

        for i in range(size[2]):
            image_slice3 = image[:,:,i,:,:]
            image_slice_norm = transform(image_slice3)
            image_norms.append(image_slice_norm.unsqueeze(2))
        image_norms = torch.cat(image_norms,dim=2)
        if use_group:
            image_norms = image_norms.unsqueeze(0)
            grouped_channels = []
            for start_channel in range(0,group_num):
                grouped_channels.append(np.arange(start_channel,(image_norms.shape[2]//group_num)*group_num,group_num))
            grouped_img = torch.cat([image_norms[:, :, channels, :, :] for channels in grouped_channels], dim=0)
            return grouped_img.cuda()
        else:
            return image_norms
