import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import normalize

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

def load_data(data_dir,name):
    """
    加载数据
    """
    images = []
    for i in range(0, 12):
        img_path = os.path.join(data_dir, f'{name}.{i}.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        # 归一化到 [0, 1]
        images.append(img.astype(np.float32) / 255.0)
    
    images = np.array(images)
    mask = cv2.imread(os.path.join(data_dir, f'{name}.mask.png'), cv2.IMREAD_GRAYSCALE)
    # 读取光源 lights.txt (假设每行是 x y z)
    lights = np.loadtxt(os.path.join(data_dir, 'lights.txt'), skiprows=1)
    
    return images, lights, mask

def visualize_normals(normals, mask):
    """
    可视化函数
    """
    # 按照公式: (N + 1) * 128
    # 此时 vis_n 的通道顺序是 [nx, ny, nz]
    vis_n = ((normals + 1.0) * 128.0).clip(0, 255).astype(np.uint8)
    
    # 处理背景：将非 mask 区域设为 0 (黑色)
    vis_n[mask == 0] = 0 
    
    return vis_n 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Photometric Stereo")
    parser.add_argument(
        "--name",
        type=str,
    )
    args = parser.parse_args()

    data_folder = f"./{args.name}"
    

    imgs, lts, mask = load_data(data_folder,args.name)
    n_map = solve_photometric_stereo(imgs, lts, mask)
        
    # 结果展示
    # 传入 mask 以便处理背景
    vis_map_rgb = visualize_normals(n_map, mask)
        
    # 保存结果
    # cv2.imwrite 期望的是 BGR 顺序,保存时需要转回 BGR
    save_bgr = cv2.cvtColor(vis_map_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(data_folder, f"{args.name}_normal_result.png"), save_bgr)
