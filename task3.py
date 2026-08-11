import cv2
import numpy as np
import os
from scipy.io import loadmat

def solve_ps_thresholding(images, lights, mask, low, high):
    """
    带阈值筛选的光度立体视觉算法
    :param images: (K, H, W) 归一化后的图像
    :param lights: (K, 3) 光源方向
    :param mask: (H, W) 掩模
    :param low: 亮度下界（剔除阴影）
    :param high: 亮度上界（剔除高光）
    """
    K, H, W = images.shape
    normals = np.zeros((H, W, 3), dtype=np.float32)

    start_idx = int(K * low)
    end_idx = int(K * high)    
    # 逐像素处理，因为每个像素留下的光源数量不同，无法直接矩阵化
    # 为了提速，我们只遍历 mask 内的像素
    rows, cols = np.where(mask > 0)
    
    for r, c in zip(rows, cols):
        # 提取该像素在所有光源下的亮度值
        I_pixel = images[:, r, c]
        
        # 找到符合 [low, high] 比例范围的索引
        sorted_idx = np.argsort(I_pixel)
        valid_idx = sorted_idx[start_idx:end_idx]

        # 至少需要 3 个光源才能求解
        if len(valid_idx) >= 3:
            L_sub = lights[valid_idx]
            I_sub = I_pixel[valid_idx]
            
            # 最小二乘求解 g = rho * n
            g, _, _, _ = np.linalg.lstsq(L_sub, I_sub, rcond=None)
            
            rho = np.linalg.norm(g)
            if rho > 0:
                normals[r, c] = g / rho
        else:
            # 如果有效光源太少，退而求其次使用全部光源计算，或设为 0
            g, _, _, _ = np.linalg.lstsq(lights, I_pixel, rcond=None)
            rho = np.linalg.norm(g)
            if rho > 0:
                normals[r, c] = g / rho
                
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


root = "DiLiGenT/pmsData/" # 指向 pmsData 文件夹
objs = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG', 'gobletPNG', 'harvestPNG', 'pot1PNG', 'pot2PNG', 'readingPNG']

if __name__ == "__main__":
    T_LOW = 0.4
    T_HIGH = 0.6
    print(f"{'Object':<10} | {'MAE (deg)':<10}")
    print("-" * 25)

    for obj in objs:
        path = os.path.join(root, obj)
        if not os.path.exists(path): continue
        
        imgs, lits, msk, gt = load_diligent_object(path)
        n_pred = solve_ps_thresholding(imgs, lits, msk,T_LOW,T_HIGH)
        
        mae = compute_angular_error(n_pred, gt, msk)
        print(f"{obj:<10} | {mae:<10.2f}")
        
        # 可视化并保存结果
        vis = ((n_pred + 1.0) * 128).clip(0, 255).astype(np.uint8)
        vis[msk == 0] = 0
        cv2.imwrite(f"{obj}_result_3.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))