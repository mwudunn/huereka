# File for helpful color operations ported to Tensorflow
# All tf operations here are batched in the first dimension
# Thanks to colormath for the original implementations https://pypi.org/project/colormath/

import numpy as np
import tensorflow as tf
from colormath.color_objects import LabColor, sRGBColor, XYZColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color

def deg2rad(deg):
    return deg * 0.017453292519943295

def rad2deg(rad):
    return rad / 0.017453292519943295

def sRGB_to_XYZ(colors):
    # Requires RGB values 0.0-1.0, NOT 0-255
    mat = np.array([(0.412424, 0.212656, 0.0193324),
                    (0.357579, 0.715158, 0.119193),
                    (0.180464, 0.0721856, 0.950444)], dtype='float32')
    colors = tf.reshape(colors, (-1,))
    mask = tf.cast(colors <= 0.04045, tf.float32)
    if_small = colors / 12.92
    if_large = tf.pow((colors + 0.055) / 1.055, 2.4)
    colors = tf.zeros(tf.shape(colors)) + mask * if_small
    colors = colors + (1 - mask) * if_large
    colors = tf.reshape(colors, (-1,3))
    colors = tf.matmul(colors, mat)
    return colors

def XYZ_to_LAB(colors):
    # Assume illuminant D65, 2deg observer
    illum = np.array([0.95047, 1.0, 1.08883], dtype='float32')
    colors = colors / illum
    CIE_E = 216.0 / 24389.0

    colors = tf.reshape(colors, (-1,))
    mask = tf.cast(colors > CIE_E, tf.float32)
    if_large = tf.pow(colors, (1. / 3.))
    if_small = (7.787 * colors) + (16. / 116.)
    colors = tf.zeros(tf.shape(colors)) + mask * if_large
    colors = colors + (1 - mask) * if_small
    colors = tf.reshape(colors, (-1,3))

    l = (116. * colors[:,1]) - 16.
    a = 500. * (colors[:,0] - colors[:,1])
    b = 200. * (colors[:,1] - colors[:,2])

    return tf.stack([l,a,b], axis=1)

def deltaE_2000(colors1, colors2):
    Kl, Kc, Kh = 1., 1., 1.
    L, a, b = tf.unstack(colors1, axis=1)

    avg_Lp = (L + colors2[:, 0]) / 2.

    C1 = tf.sqrt(tf.reduce_sum(tf.pow(colors1[:,1:], 2.), axis=1))
    C2 = tf.sqrt(tf.reduce_sum(tf.pow(colors2[:,1:], 2.), axis=1))

    avg_C1_C2 = (C1 + C2) / 2.

    G = 0.5 * (1. - tf.sqrt(tf.pow(avg_C1_C2, 7.) / (tf.pow(avg_C1_C2, 7.) + tf.pow(25., 7.))))

    a1p = (1.0 + G) * a
    a2p = (1.0 + G) * colors2[:, 1]

    C1p = tf.sqrt(tf.pow(a1p, 2.) + tf.pow(b, 2.))
    C2p = tf.sqrt(tf.pow(a2p, 2.) + tf.pow(colors2[:, 2], 2.))

    avg_C1p_C2p = (C1p + C2p) / 2.

    h1p = rad2deg(tf.atan2(b, a1p))
    h1p = h1p + tf.cast(h1p < 0., tf.float32) * 360.

    h2p = rad2deg(tf.atan2(colors2[:,2], a2p))
    h2p = h2p + tf.cast(h2p < 0., tf.float32) * 360.

    avg_Hp = ((tf.cast(tf.abs(h1p - h2p) > 180., tf.float32) * 360.) + h1p + h2p) / 2.0

    T = 1. - 0.17 * tf.cos(deg2rad(avg_Hp - 30.)) + \
        0.24 * tf.cos(deg2rad(2 * avg_Hp)) + \
        0.32 * tf.cos(deg2rad(3 * avg_Hp + 6.)) - \
        0.2 * tf.cos(deg2rad(4 * avg_Hp - 63.))

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + tf.cast(tf.abs(diff_h2p_h1p) > 180., tf.float32) * 360.
    delta_hp = delta_hp - tf.cast(h2p > h1p, tf.float32) * 720.

    delta_Lp = colors2[:, 0] - L
    delta_Cp = C2p - C1p
    delta_Hp = 2. * tf.sqrt(C2p * C1p) * tf.sin(deg2rad(delta_hp) / 2.0)

    S_L = 1 + ((0.015 * tf.pow(avg_Lp - 50., 2.)) / tf.sqrt(20. + tf.pow(avg_Lp - 50, 2.)))
    S_C = 1 + 0.045 * avg_C1p_C2p
    S_H = 1 + 0.015 * avg_C1p_C2p * T

    delta_ro = 30. * tf.exp(-(tf.pow(((avg_Hp - 275.) / 25.), 2.)))
    R_C = tf.sqrt((tf.pow(avg_C1p_C2p, 7.)) / (tf.pow(avg_C1p_C2p, 7.) + tf.pow(25., 7.)))
    R_T = -2. * R_C * tf.sin(2. * deg2rad(delta_ro))

    return tf.sqrt(
        tf.pow(delta_Lp / (S_L * Kl), 2.) +
        tf.pow(delta_Cp / (S_C * Kc), 2.) +
        tf.pow(delta_Hp / (S_H * Kh), 2.) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))


def main():
    """
    Test operations
    """
    tf.compat.v1.enable_eager_execution()

    # Test sRGB to XYZ
    orange_sRGB = sRGBColor(252, 127, 3, is_upscaled=True)
    red_sRGB = sRGBColor(204, 39, 2, is_upscaled=True)
    orange_XYZ = convert_color(orange_sRGB, XYZColor)
    red_XYZ = convert_color(red_sRGB, XYZColor)

    print('orange XYZ ref:', orange_XYZ)
    print('red XYZ ref:', red_XYZ)

    colors_sRGB = np.array([(252, 127, 3), (204, 39, 2)], dtype='float32')
    colors_sRGB /= 255.
    colors_sRGB_tf = tf.constant(colors_sRGB)
    colors_XYZ_tf = sRGB_to_XYZ(colors_sRGB_tf)
    colors_XYZ = colors_XYZ_tf.numpy()

    print('orange XYZ tf:', colors_XYZ[0])
    print('red XYZ tf:', colors_XYZ[1])

    # Test XYZ to LAB
    orange_LAB = convert_color(orange_XYZ, LabColor)
    red_LAB = convert_color(red_XYZ, LabColor)

    print('orange LAB ref:', orange_LAB)
    print('red LAB ref:', red_LAB)

    colors_LAB_tf = XYZ_to_LAB(colors_XYZ_tf)
    colors_LAB = colors_LAB_tf.numpy()

    print('orange LAB tf:', colors_LAB[0])
    print('red LAB tf:', colors_LAB[1])

    blue_grey_sRGB = sRGBColor(74, 81, 97, is_upscaled=True)
    hot_pink_SRGB = sRGBColor(240, 113, 240, is_upscaled=True)
    blue_grey_LAB = convert_color(blue_grey_sRGB, LabColor)
    hot_pink_LAB = convert_color(hot_pink_SRGB, LabColor)
    diff1 = delta_e_cie2000(orange_LAB, red_LAB)
    diff2 = delta_e_cie2000(blue_grey_LAB, hot_pink_LAB)

    print('deltaE ref:', diff1, diff2)

    colors1_sRGB = np.array([(252, 127, 3), (74, 81, 97)], dtype='float32') / 255.
    colors2_sRGB = np.array([(204, 39, 2), (240, 113, 240)], dtype='float32') / 255.
    colors1_sRGB_tf = tf.constant(colors1_sRGB)
    colors2_sRGB_tf = tf.constant(colors2_sRGB)
    colors1_XYZ_tf = sRGB_to_XYZ(colors1_sRGB_tf)
    colors2_XYZ_tf = sRGB_to_XYZ(colors2_sRGB_tf)
    colors1_LAB_tf = XYZ_to_LAB(colors1_XYZ_tf)
    colors2_LAB_tf = XYZ_to_LAB(colors2_XYZ_tf)
    diff_tf = deltaE_2000(colors1_LAB_tf, colors2_LAB_tf)

    print('deltaE tf:', diff_tf.numpy())


if __name__ == '__main__':
    main()