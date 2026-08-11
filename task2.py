import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import normalize
from scipy.io import loadmat

def solve_photometric_stereo(images, lights, mask):
    """
    使用最小二乘法求解光度立体视觉 
    images: 形状为 (K, H, W) 的数组，K为光源数量
    lights: 形状为 (K, 3) 的光源方向矩阵
    mask: 形状为 (H, W) 的掩模，255表示物体，0表示背景
    normals (H, W, 3) 归一化后的法线图
    """
    K, H, W = images.shape

    # 预计算光源矩阵的伪逆，加速计算: (L^T * L)^-1 * L^T
    # 形状为 (3, K)
    L_inv = np.linalg.pinv(lights)

    # 为了加速，我们将图像展平，只处理 mask 内的像素
    mask_flat = mask.reshape(-1)
    images_flat = images.reshape(K, -1)
    
    # 获取物体像素的索引
    object_indices = np.where(mask_flat > 0)[0]

    # 提取物体像素的亮度值 (K, N_pixels)
    I = images_flat[:, object_indices]
    
    # # 最小二乘求解 g = L_inv * I -> (3, N_pixels)
    g = np.dot(L_inv, I)

    # 计算反照率 rho (1, N_pixels)
    rho = np.linalg.norm(g, axis=0, keepdims=True)

    # 防止除以0
    rho[rho == 0] = 1

    # 使用 sklearn 的 normalize 对每列向量进行归一化，得到单位法线 n (3, N_pixels)
    n = normalize(g, axis=0)
    
    # 将计算结果填回原图形状
    normals_flat = np.zeros((3, H * W), dtype=np.float32)
    normals_flat[:, object_indices] = n

    # 转置回 (H, W, 3)
    normals = normals_flat.T.reshape(H, W, 3)
    
    return normals

def compute_angular_error(n_pred, n_gt, mask):
    """计算平均角度误差 """
    # 只取 mask 内的像素
    indices = np.where(mask > 0)
    n1 = n_pred[indices] # (N, 3)
    n2 = n_gt[indices]   # (N, 3)
    
    # 逐行做内积: np.sum(a*b, axis=1)
    # 使用 clip 确保由于浮点数误差导致的点积不会超过 [-1, 1] 范围
    dot_product = np.sum(n1 * n2, axis=1).clip(-1, 1)
    
    # 计算角度
    angular_err = np.real((180.0 * np.arccos(dot_product)) / np.pi)
    
    return np.mean(angular_err)

def load_diligent_object(obj_path):
    """专门适配 DiLiGenT 数据结构的加载函数"""
    # 加载光源方向 
    l_dirs = np.loadtxt(os.path.join(obj_path, 'light_directions.txt'))
    # 加载光源强度 
    l_ints = np.loadtxt(os.path.join(obj_path, 'light_intensities.txt'))
    # 加载文件名列表
    with open(os.path.join(obj_path, 'filenames.txt'), 'r') as f:
        filenames = [line.strip() for line in f.readlines()]
    
    # 加载 Mask
    mask = cv2.imread(os.path.join(obj_path, 'mask.png'), 0)
    
    # 加载并归一化图像
    images = []
    for i, fname in enumerate(filenames):
        img_path = os.path.join(obj_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        # 1. 关键：先除以 65535.0，将 16-bit 映射到 [0, 1]
        img = img / 65535.0
        
        # 2. 转灰度 (按照 DiLiGenT 推荐，可以取平均或用特定通道)
        img_gray = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0
        
        # 3. 除以光强进行归一化
        avg_int = np.mean(l_ints[i]) 
        img_normalized = img_gray / avg_int
        
        images.append(img_normalized)
        
    images = np.array(images)
    
    # 6. 加载真值
    gt_data = loadmat(os.path.join(obj_path, 'Normal_gt.mat'))
    n_gt = gt_data['Normal_gt'].astype(np.float32)
    
    return images, l_dirs, mask, n_gt


root = "DiLiGenT/pmsData/" # 指向你的 pmsData 文件夹
objs = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG', 'gobletPNG', 'harvestPNG', 'pot1PNG', 'pot2PNG', 'readingPNG']

if __name__ == "__main__":
    print(f"{'Object':<10} | {'MAE (deg)':<10}")
    print("-" * 25)

    for obj in objs:
        path = os.path.join(root, obj)
        if not os.path.exists(path): continue
        
        imgs, lits, msk, gt = load_diligent_object(path)
        n_pred = solve_photometric_stereo(imgs, lits, msk)
        
        mae = compute_angular_error(n_pred, gt, msk)
        print(f"{obj:<10} | {mae:<10.2f}")
        
        # 可视化并保存结果
        vis = ((n_pred + 1.0) * 128).clip(0, 255).astype(np.uint8)
        vis[msk == 0] = 0
        cv2.imwrite(f"{obj}_result.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))