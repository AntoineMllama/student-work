import cv2
import numpy as np


def load_img(load: str) -> np.ndarray:
    """

    :param load: str
    :return: np.ndarray(width, height, 3)

    Load image from file.
    """
    return cv2.imread(load)[:, :, [2, 1, 0]]  # OpenCV charge en BGR, on convertit au passage en RGB


def save_img(where: str, img: np.array) -> None:  # img est sous format RGB
    """

    :param where: str
    :param img: np.ndarray(width, height, 3)
    :return: None

    Save image to file.
    """
    img = img[:, :, [2, 1, 0]]
    cv2.imwrite(where, img)
    return


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """

    :param img: np.ndarray(x, y, 3)
    :return: np.ndarray(x, y)

    Converting RGB image to gray image
    """
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """

    :param img: np.ndarray(x, y)
    :return: np.ndarray(x, y, 3)

    Converting gray image to RGB image
    """
    return np.stack((img, img, img), axis=-1)


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    img_ycbcr = np.zeros_like(img, dtype=np.float32)
    
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    
    img_ycbcr[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    img_ycbcr[:, :, 1] = 128 + (-0.168736 * R - 0.331264 * G + 0.5 * B)
    img_ycbcr[:, :, 2] = 128 + (0.5 * R - 0.418688 * G - 0.081312 * B)
    
    return img_ycbcr


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    img_rgb = np.zeros_like(img, dtype=np.float32)
    
    Y = img[:, :, 0].astype(np.float32)
    Cb = img[:, :, 1].astype(np.float32) - 128
    Cr = img[:, :, 2].astype(np.float32) - 128
    
    img_rgb[:, :, 0] = Y + 1.402 * Cr
    img_rgb[:, :, 1] = Y - 0.344136 * Cb - 0.714136 * Cr
    img_rgb[:, :, 2] = Y + 1.772 * Cb
    
    np.clip(img_rgb, 0, 255, out=img_rgb)
    
    return img_rgb.astype(np.uint8)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    cmax = np.max(img, axis=-1)
    cmin = np.min(img, axis=-1)

    delta_c = cmax - cmin
    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax

    mask = delta_c != 0
    idx = (cmax == r) & mask
    h[idx] = 60 * (((g[idx] - b[idx]) / delta_c[idx]) % 6)
    idx = (cmax == g) & mask
    h[idx] = 60 * (((b[idx] - r[idx]) / delta_c[idx]) + 2)
    idx = (cmax == b) & mask
    h[idx] = 60 * (((r[idx] - g[idx]) / delta_c[idx]) + 4)

    s[mask] = delta_c[mask] / cmax[mask]

    hsv = np.stack((h, s, v), axis=-1)

    return hsv.astype(np.float32)


def hsv_to_rgb(img: np.ndarray) -> np.ndarray:
    img.astype(np.float32)
    h, s, v = img[..., 0], img[..., 1], img[..., 2]

    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c

    rp = np.zeros_like(h)
    gp = np.zeros_like(h)
    bp = np.zeros_like(h)

    idx = (h < 60)
    rp[idx] = c[idx]
    gp[idx] = x[idx]
    bp[idx] = 0

    idx = (h >= 60) & (h < 120)
    rp[idx] = x[idx]
    gp[idx] = c[idx]
    bp[idx] = 0

    idx = (h >= 120) & (h < 180)
    rp[idx] = 0
    gp[idx] = c[idx]
    bp[idx] = x[idx]

    idx = (h >= 180) & (h < 240)
    rp[idx] = 0
    gp[idx] = x[idx]
    bp[idx] = c[idx]

    idx = (h >= 240) & (h < 300)
    rp[idx] = x[idx]
    gp[idx] = 0
    bp[idx] = c[idx]

    idx = (h >= 300) & (h < 360)
    rp[idx] = c[idx]
    gp[idx] = 0
    bp[idx] = x[idx]

    r = (rp + m) * 255
    g = (gp + m) * 255
    b = (bp + m) * 255

    rgb = np.stack((r, g, b), axis=-1)

    return rgb.astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0] / 255.0
    g = rgb[:, :, 1] / 255.0
    b = rgb[:, :, 2] / 255.0

    r = np.where(r > 0.04045, np.power((r + 0.055) / 1.055, 2.4), r / 12.92)
    g = np.where(g > 0.04045, np.power((g + 0.055) / 1.055, 2.4), g / 12.92)
    b = np.where(b > 0.04045, np.power((b + 0.055) / 1.055, 2.4), b / 12.92)

    r *= 100
    g *= 100
    b *= 100

    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    x /= 95.047
    y /= 100.0
    z /= 108.883

    x = np.where(x > 0.008856, np.power(x, 1/3.0), (7.787 * x) + (16.0 / 116.0))
    y = np.where(y > 0.008856, np.power(y, 1/3.0), (7.787 * y) + (16.0 / 116.0))
    z = np.where(z > 0.008856, np.power(z, 1/3.0), (7.787 * z) + (16.0 / 116.0))

    L = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    lab = np.stack([L, a, b], axis=-1)
    return lab


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    reference_white = np.array([95.047, 100.0, 108.883])

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    Y = (L + 16) / 116
    X = a / 500 + Y
    Z = Y - b / 200

    epsilon = 0.008856
    kappa = 903.3

    X = np.where(X ** 3 > epsilon, X ** 3, (X - 16 / 116) / 7.787)
    Y = np.where(Y ** 3 > epsilon, Y ** 3, (Y - 16 / 116) / 7.787)
    Z = np.where(Z ** 3 > epsilon, Z ** 3, (Z - 16 / 116) / 7.787)

    img_xyz = np.stack([X, Y, Z], axis=-1) * reference_white
    return img_xyz


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz / 100.0

    transformation_matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])
    rgb = np.dot(xyz, transformation_matrix.T)

    rgb = np.clip(rgb, 0, None)

    rgb = np.where(rgb > 0.0031308, 1.055 * (rgb ** (1 / 2.4)) - 0.055, 12.92 * rgb)

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    return rgb
