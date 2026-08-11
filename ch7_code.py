import numpy as np
import cv2
import glob
from sklearn.preprocessing import normalize
# Read light directions
Lt = np.loadtxt('data/cat/lights.txt')  # Transpose of L
# Read images
M = []
for fname in sorted(glob.glob('data/cat/*.png')):
    im = cv2.imread(fname, 0)
    if M == []:
        height, width = im.shape
        M = im.reshape((-1, 1))
    else:
        M = np.append(M, im.reshape((-1,1)), axis=1)
# Photometric stereo computation (least-squares)
# M = NL <-> M^T = L^T N^T.
N = np.linalg.lstsq(Lt, M.T)[0].T
N = normalize(N, axis=1)  # normalize to account for diffuse reflectance
# Visualization
N = np.reshape(N, (height, width, 3))  # Reshape to image coordinates
N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
N = (N + 1.0) / 2.0  # Rescale
cv2.imshow('normal map', N)
cv2.waitKey()
cv2.destroyAllWindows()
