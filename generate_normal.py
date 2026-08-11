import numpy as np
import scipy.io as sio
import imageio.v2 as imageio


# 参数

size = 340        # 图像大小
radius = 100        # 球半径

# 输出文件
mat_name = "Normal_gt.mat"
png_name = "Normal_gt.png"
mask_name = "sphere_mask.png"

# 创建坐标网格

cx = cy = size // 2

y, x = np.mgrid[0:size, 0:size]

x = (x - cx) / radius
y = (y - cy) / radius

# 球 mask
mask = x**2 + y**2 <= 1.0

# 计算球面法向量

z = np.zeros_like(x)
z[mask] = np.sqrt(1.0 - x[mask]**2 - y[mask]**2)

# normal: (H, W, 3)
normal = np.zeros((size, size, 3), dtype=np.float32)

# 注意：
# 图像坐标 y 向下，因此这里取负号更符合视觉习惯
normal[..., 0] = x
normal[..., 1] = -y
normal[..., 2] = z

# 单位化
norm = np.linalg.norm(normal, axis=2, keepdims=True)
norm[norm == 0] = 1
normal = normal / norm

# 保存 .mat

sio.savemat(mat_name, {
    "Normal_gt": normal
})

print(f"Saved {mat_name}")


# 生成 normal 可视化图
# [-1,1] -> [0,255]

vis = (normal + 1.0) / 2.0
vis = (vis * 255).astype(np.uint8)

# mask 外设为黑色
vis[~mask] = 0

imageio.imwrite(png_name, vis)

print(f"Saved {png_name}")


# 保存 mask

mask_img = np.zeros((size, size), dtype=np.uint8)
mask_img[mask] = 255

imageio.imwrite(mask_name, mask_img)

print(f"Saved {mask_name}")