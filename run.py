import cv2
import numpy as np


def convert_to_gray_scale(image):
    R, G, B = 0.21, 0.72, 0.07

    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    gray_scale_image = R * r_channel + G * g_channel + B * b_channel

    gray_scale_image = gray_scale_image.astype("uint8")

    return gray_scale_image


def convolution_2d(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width), "float32")

    for i in range(output_height):
        for j in range(output_width):
            image_region = image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(image_region * kernel)

    return output


def apply_gaussian_filter(image):
    gaussian_filter_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], "float32") / 16

    return convolution_2d(image, gaussian_filter_3x3)


def calc_gradient(image):
    Mx = np.array([[0, 1], [-1, 0]], "float32")
    My = np.array([[1, 0], [0, -1]], "float32")

    Ix = convolution_2d(image, Mx)
    Iy = convolution_2d(image, My)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(G, theta):
    height, width = G.shape
    after_nm_suppression = np.zeros((height, width), "float32")

    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            try:
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                if (G[i, j] >= q) and (G[i, j] >= r):
                    after_nm_suppression[i, j] = G[i, j]
                else:
                    after_nm_suppression[i, j] = 0
            except:
                pass
    
    return after_nm_suppression


image = cv2.imread("a_image.webp")

gray_scale_image = convert_to_gray_scale(image)

gf_applied_image = apply_gaussian_filter(gray_scale_image)

G, theta = calc_gradient(gf_applied_image)

after_nm_suppression = non_max_suppression(G, theta)

canny_edge_image = np.zeros(after_nm_suppression.shape, "uint8")
canny_edge_image[np.where(after_nm_suppression >= 45)] = np.uint8(255)
canny_edge_image[np.where((15 < after_nm_suppression) & (after_nm_suppression < 45))] = np.uint8(15)

print("Original image: " + str(image.shape))
print("Canny edge image: " + str(canny_edge_image.shape))

cv2.imwrite("b_gray_scale.jpg", gray_scale_image)
cv2.imwrite("c_gf_applied_image.jpg", gf_applied_image)
cv2.imwrite("d_gradient.jpg", G)
cv2.imwrite("e_after_nm_suppression.jpg", after_nm_suppression)
cv2.imwrite("f_canny_edge_image.jpg", canny_edge_image)
