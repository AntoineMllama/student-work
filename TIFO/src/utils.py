import numpy as np
from tqdm import tqdm


def local_neighbour(img: np.ndarray, x: int, y: int, kernel_size: int) -> np.ndarray:
    """

    :param img: np.ndarray(x, y)
    :param x: int
    :param y: int
    :param kernel_size: int
    :return: np.ndarray(z)

    Returns neighbour image at position (x, y)
    """
    kernel_half = kernel_size // 2
    x_min = max(0, x - kernel_half)
    x_max = min(img.shape[0], x + kernel_half + 1)
    y_min = max(0, y - kernel_half)
    y_max = min(img.shape[1], y + kernel_half + 1)
    return img[x_min:x_max, y_min:y_max]


def local_mean_var(img: np.ndarray, size: int) -> (np.ndarray, np.ndarray):
    """

    :param img: np.ndarray(x, y)
    :param size: int
    :return: (np.ndarray, np.ndarray)

    Returns mean and variance image at all position
    """
    width = img.shape[0]
    height = img.shape[1]
    local_mean = np.zeros((width, height), dtype=float)
    local_var = np.zeros((width, height), dtype=float)
    for j in tqdm(range(height), desc="Calculating Local Mean and Variance"):  # tqdm permet de faire une progress bar
        for i in range(width):
            neighbour = local_neighbour(img, i, j, size)
            local_mean[i, j] = np.mean(neighbour)
            local_var[i, j] = np.var(neighbour)
    return local_mean, local_var


def local_mean_var_rgb(img: np.ndarray, size: int) ->\
        ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """

    :param img: np.ndarray(x, y)
    :param size: int
    :return: ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray))

    Returns mean and variance image_rgb at all position
    """
    width = img.shape[0]
    height = img.shape[1]

    local_mean_r = np.zeros((width, height), dtype=float)
    local_var_r = np.zeros((width, height), dtype=float)

    local_mean_g = np.zeros((width, height), dtype=float)
    local_var_g = np.zeros((width, height), dtype=float)

    local_mean_b = np.zeros((width, height), dtype=float)
    local_var_b = np.zeros((width, height), dtype=float)

    for j in tqdm(range(height), desc="Calculating RGB Local Mean and Variance"):
        for i in range(width):
            neighbour = local_neighbour(img, i, j, size)
            neighbour_r, neighbour_g, neighbour_b = neighbour[:, :, 0], neighbour[:, :, 1], neighbour[:, :, 2]

            local_mean_r[i, j] = np.mean(neighbour_r)
            local_var_r[i, j] = np.var(neighbour_r)

            local_mean_g[i, j] = np.var(neighbour_g)
            local_var_g[i, j] = np.var(neighbour_g)

            local_mean_b[i, j] = np.var(neighbour_b)
            local_var_b[i, j] = np.var(neighbour_b)

    return (local_mean_r, local_var_r), (local_mean_g, local_var_g), (local_mean_b, local_var_b)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 25.0) -> np.ndarray:
    """

    :param image: np.ndarray(x, y)
    :param mean: float
    :param std: float
    :return: np.ndarray(x, y)

    Returns Gaussian noise image
    """
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


def add_poisson_noise(image: np.ndarray, scale: int = 50) -> np.ndarray:
    """

    :param image: np.ndarray(x, y)
    :param scale: int
    :return: np.ndarray(x, y)

    Returns Poisson noise image
    """
    normalized_image = image / 255.0
    adjusted_image = normalized_image * 30
    noisy_image = np.random.poisson(adjusted_image) / scale
    noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    return noisy_image
