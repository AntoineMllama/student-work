import image
import utils

import numpy as np

# PATH Linux
# IN: str = 'image/img.png'
# OUT: str = 'image/out.png'
# BLUR: str = 'image/blur.png'

# PATH Windows
IN: str = 'C:\\Users\\Antoine MENARD\\Documents\\cours\\ING2\\tifo\\image\\lama.jpg'
OUT: str = 'C:\\Users\\Antoine MENARD\\Documents\\cours\\ING2\\tifo\\image\\out.png'
BLUR: str = 'C:\\Users\\Antoine MENARD\\Documents\\cours\\ING2\\tifo\\image\\blur.png'


def wiener_gray(img_path: str, out_path: str,  kernel_size: int = 3, blur_save: bool = False, blur_path: str = None) -> None:
    """

    :param img_path: str
    :param out_path: str
    :param kernel_size: int
    :param blur_save: bool
    :param blur_path: str
    :return: None

    Compute Wiener filter
    """
    if blur_save and blur_path is None:
        print("ERROR: blur_save is True but blur_path is None")
        return

    base_img = image.load_img(img_path)
    gray_img = image.rgb_to_gray(base_img)

    blur_image = utils.add_gaussian_noise(gray_img, mean=0, std=200)  # blur avec un bruit gaussian
    # blur_image = utils.add_poisson_noise(gray_img)  # blur avec un bruit de poisson
    # blur_image = gray_img.copy()  # si l'image est deja blur

    my_mean, my_var = utils.local_mean_var(blur_image, kernel_size)

    noise_variance = np.mean(my_var)
    wiener_res = my_mean + (np.maximum(my_var - noise_variance, 0) / (my_var + noise_variance)) * (blur_image - my_mean)
    wiener_res = image.gray_to_rgb(wiener_res)
    blur_image = image.gray_to_rgb(blur_image)

    image.save_img(out_path, wiener_res)
    if blur_save:
        image.save_img(blur_path, blur_image)
    return


def wiener_trois_caneaux(base_img: np.ndarray, mode: str = None, kernel_size: int = 3) -> (np.ndarray, np.ndarray):
    """

    :param base_img: np.ndarray(x, y, 3)
    :param mode: str
    :param kernel_size: int = 3
    :return: (np.ndarray(x, y, 3), np.ndarray(x, y, 3))
    """
    img_a = base_img[:, :, 0]
    img_b = base_img[:, :, 1]
    img_c = base_img[:, :, 2]

    blur_image_a = utils.add_gaussian_noise(img_a, mean=0, std=200)
    blur_image_b = utils.add_gaussian_noise(img_b, mean=0, std=200)
    blur_image_c = utils.add_gaussian_noise(img_c, mean=0, std=200)

    all_blur = np.stack((blur_image_a, blur_image_b, blur_image_c), axis=-1)

    if mode == "hsv":
        all_blur = image.rgb_to_hsv(all_blur)
    elif mode == "YCbCr":
        all_blur = image.rgb_to_ycbcr(all_blur)
    elif mode == "Lab":
        all_blur = image.rgb_to_lab(all_blur)

    blur_image_a = all_blur[:, :, 0]
    blur_image_b = all_blur[:, :, 1]
    blur_image_c = all_blur[:, :, 2]

    my_mean_a, my_var_a = utils.local_mean_var(blur_image_a, kernel_size)
    my_mean_b, my_var_b = utils.local_mean_var(blur_image_b, kernel_size)
    my_mean_c, my_var_c = utils.local_mean_var(blur_image_c, kernel_size)

    noise_variance_a = np.mean(my_var_a)
    noise_variance_b = np.mean(my_var_b)
    noise_variance_c = np.mean(my_var_c)

    dev_a = (np.maximum(my_var_a - noise_variance_a, 0) / (my_var_a + noise_variance_a))
    dev_b = (np.maximum(my_var_b - noise_variance_b, 0) / (my_var_b + noise_variance_b))
    dev_c = (np.maximum(my_var_c - noise_variance_c, 0) / (my_var_c + noise_variance_c))

    wiener_res_a = my_mean_a + np.nan_to_num(dev_a, nan=0.0) * (blur_image_a - my_mean_a)
    wiener_res_b = my_mean_b + np.nan_to_num(dev_b, nan=0.0) * (blur_image_b - my_mean_b)
    wiener_res_c = my_mean_c + np.nan_to_num(dev_c, nan=0.0) * (blur_image_c - my_mean_c)

    return np.stack((wiener_res_a, wiener_res_b, wiener_res_c), axis=-1), all_blur


def wiener_rgb_all(img_path: str, out_path: str,  kernel_size: int = 3, blur_save: bool = False, blur_path: str = None) -> None:
    """

    :param img_path: str
    :param out_path: str
    :param kernel_size: int
    :param blur_save: bool
    :param blur_path: str
    :return: None

    Compute Wiener RGB filter
    """
    if blur_save and blur_path is None:
        print("ERROR: blur_save is True but blur_path is None")
        return
    base_img = image.load_img(img_path)

    blur_image = utils.add_gaussian_noise(base_img, mean=0, std=200)

    r, g, b = utils.local_mean_var_rgb(img=blur_image, size=kernel_size)  # x[0] = mean, x[1] = var

    noise_variance_r = np.mean(r[1])
    noise_variance_g = np.mean(g[1])
    noise_variance_b = np.mean(b[1])

    wiener_res_r = r[0] + (np.maximum(r[1] - noise_variance_r, 0) / (r[1] + noise_variance_r)) * (blur_image[:, :, 0] - r[0])
    wiener_res_g = g[0] + (np.maximum(g[1] - noise_variance_g, 0) / (g[1] + noise_variance_g)) * (blur_image[:, :, 1] - g[0])
    wiener_res_b = b[0] + (np.maximum(b[1] - noise_variance_b, 0) / (b[1] + noise_variance_b)) * (blur_image[:, :, 2] - b[0])

    fusion = np.stack((wiener_res_r, wiener_res_g, wiener_res_b), axis=-1)

    image.save_img(where=out_path, img=fusion)
    if blur_save:
        image.save_img(blur_path, blur_image)
    return


def wiener_rgb(img_path: str, out_path: str,  kernel_size: int = 3, blur_save: bool = False, blur_path: str = None) -> None:
    """

    :param img_path: str
    :param out_path: str
    :param kernel_size: int
    :param blur_save: bool
    :param blur_path: str
    :return: None

    Compute Wiener RGB filter
    """
    if blur_save and blur_path is None:
        print("ERROR: blur_save is True but blur_path is None")
        return
    base_img = image.load_img(img_path)
    fusion, blur_image = wiener_trois_caneaux(base_img, kernel_size=kernel_size)
    image.save_img(where=out_path, img=fusion)
    if blur_save:
        image.save_img(blur_path, blur_image)
    return


def wiener_hsv(img_path: str, out_path: str,  kernel_size: int = 3, blur_save: bool = False, blur_path: str = None) -> None:
    if blur_save and blur_path is None:
        print("ERROR: blur_save is True but blur_path is None")
        return
    base_img = image.load_img(img_path)
    fusion, blur_image = wiener_trois_caneaux(base_img, mode='hsv', kernel_size=kernel_size)
    hsv_img = image.hsv_to_rgb(fusion)
    image.save_img(where=out_path, img=hsv_img)
    if blur_save:
        blur_image = image.hsv_to_rgb(blur_image)
        image.save_img(blur_path, blur_image)
    return


def wiener_ycbcr(img_path: str, out_path: str,  kernel_size: int = 3, blur_save: bool = False, blur_path: str = None) -> None:
    """
    :param img_path: str
    :param out_path: str
    :param kernel_size: int
    :param blur_save: bool
    :param blur_path: str
    :return: None

    Compute Wiener filter
    """
    if blur_save and blur_path is None:
        print("ERROR: blur_save is True but blur_path is None")
        return

    base_img = image.load_img(img_path)
    fusion, blur_image = wiener_trois_caneaux(base_img, mode="YCbCr", kernel_size=kernel_size)

    ycbcr_img = image.ycbcr_to_rgb(fusion)
    image.save_img(where=out_path, img=ycbcr_img)
    if blur_save:
        blur_image = image.ycbcr_to_rgb(blur_image)
        image.save_img(blur_path, blur_image)
    return


def wiener_lab(img_path: str, out_path: str, kernel_size: int = 3, blur_save: bool = False, blur_path: str = None) -> None:
    """
    :param img_path: str
    :param out_path: str
    :param kernel_size: int
    :param blur_save: bool
    :param blur_path: str
    :return: None

    Compute Wiener filter
    """
    if blur_save and blur_path is None:
        print("ERROR: blur_save is True but blur_path is None")
        return
    base_img = image.load_img(img_path)
    fusion, blur_image = wiener_trois_caneaux(base_img, mode="Lab", kernel_size=kernel_size)

    lab_img = image.lab_to_rgb(fusion)
    image.save_img(where=out_path, img=lab_img)
    if blur_save:
        blur_image = image.lab_to_rgb(blur_image)
        image.save_img(blur_path, blur_image)
    return


if __name__ == '__main__':
    wiener_lab(IN, OUT, blur_save=True, blur_path=BLUR, kernel_size=7)
