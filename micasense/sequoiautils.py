#!/usr/bin/env python
# coding: utf-8
"""
Sequoia image processing utilities

Author: Rasmus Fenger-Nielsen (rasmusfenger@gmail.com)
Date: January 2019

Part of the code has been written by other author (see relevant sections below).
"""

import cv2
import numpy as np
import math
import datetime
import pytz
import base64
import struct

def sequoia_irradiance(meta, imageRaw, vignetteCorrection=True):

    if vignetteCorrection:
        xDim = imageRaw.shape[1]
        yDim = imageRaw.shape[0]
        V = vignette_correction(meta, xDim, yDim)
        imageRaw = imageRaw/V
    else:
        print ('vignette_correction not done')
        V = ''

    # get radiometric calibration factors (Sequoia Firmware Version 1.2.1 or later)
    sensorModel = meta.get_item('XMP:SensorModel').split(',')
    A = float(sensorModel[0])
    B = float(sensorModel[1])
    C = float(sensorModel[2])

    fNumber = meta.get_item('EXIF:FNumber')
    expTime = meta.get_item('EXIF:ExposureTime')
    gain = meta.get_item('EXIF:ISO')

    # make the calculation (for details, see application note: SEQ AN 01)
    I = fNumber ** 2 * (imageRaw - B) / (A * expTime * gain + C)

    return I, V

########################################################################################################################
# Vignette correction
# Python code is written by seanmcleod and modified by Rasmus Fenger-Nielsen
# Original code is available here:
# https://forum.developer.parrot.com/t/vignetting-correction-sample-code/5614

def vignette_correction(meta, xDim, yDim):
    polynomial2DName = meta.get_item('XMP:VignettingPolynomial2DName')
    polynomial2D = meta.get_item('XMP:VignettingPolynomial2D')
    poly = build_powers_coefficients(polynomial2DName, polynomial2D)
    vignette_factor = np.ones((yDim, xDim), dtype=np.float32)
    for y in range(0, yDim):
        for x in range(0, xDim):
            vignette_factor[y, x] = vignetting(poly, float(x) / xDim, float(y) / yDim)
    return vignette_factor

def build_powers_coefficients(powers, coefficients):
    '''
    :return: List of tuples of the form (n, m, coefficient)
    '''
    powers_coefficients = []
    power_items = powers.split(',')
    coefficient_items = coefficients.split(',')
    for i in range(0, len(power_items), 2):
        powers_coefficients.append((int(power_items[i]), int(power_items[i + 1]), float(coefficient_items[int(i / 2)])))
    return powers_coefficients

def vignetting(powers_coefficients, x, y):
    value = 0.0
    for entry in powers_coefficients:
        value = value + entry[2] * math.pow(x, entry[0]) * math.pow(y, entry[1])
    return value

########################################################################################################################
# Calculate sunshine sensor irradiance
# The code is written by Yu-Hsuan Tu (https://github.com/dobedobedo/Parrot_Sequoia_Image_Handler/tree/master/Modules)
# The code has been modified by Rasmus Fenger-Nielsen

def GetTimefromStart(meta):
    Time = datetime.datetime.strptime(meta.get_item('Composite:SubSecCreateDate'), "%Y:%m:%d %H:%M:%S.%f")
    Time_UTC = pytz.utc.localize(Time, is_dst=False)
    duration = datetime.timedelta(hours=Time_UTC.hour,
                                  minutes=Time_UTC.minute,
                                  seconds=Time_UTC.second,
                                  microseconds=Time_UTC.microsecond)
    return duration

def GetSunIrradiance(meta):
    encoded = meta.get_item('XMP:IrradianceList')
    # decode the string
    data = base64.standard_b64decode(encoded)

    # ensure that there's enough data QHHHfff
    #print (len(data))
    assert len(data) % 28 == 0

    # determine how many datasets there are
    count = len(data) // 28

    # unpack the data as uint64, uint16, uint16, uint16, uint16, float, float, float
    result = []
    for i in range(count):
        index = 28 * i
        s = struct.unpack('<QHHHHfff', data[index:index + 28])
        result.append(s)

    CreateTime = GetTimefromStart(meta)
    timestamp = []
    for measurement in result:
        q, r = divmod(measurement[0], 1000000)
        timestamp.append(abs(datetime.timedelta(seconds=q, microseconds=r) - CreateTime))
    TargetIndex = timestamp.index(min(timestamp))
    count = result[TargetIndex][1]
    gain = result[TargetIndex][3]
    exposuretime = result[TargetIndex][4]
    Irradiance = float(count) / (gain * exposuretime)
    return Irradiance

########################################################################################################################
# Correct fisheye lens distortion.
# The code is originally written in Matlab by muzammil360 (https://github.com/muzammil360/SeqUDR)
# The code has been reprogrammed in Python by Rasmus Fenger-Nielsen

def correct_lens_distortion_sequoia(meta, image):
    # get lens distortion parameters
    FisheyePoly = np.array(meta.get_item('XMP:FisheyePolynomial').split(',')).astype(np.float)
    p0 = FisheyePoly[0]
    p1 = FisheyePoly[1]
    p2 = FisheyePoly[2]
    p3 = FisheyePoly[3]

    FisheyeAffineMat = np.array(meta.get_item('XMP:FisheyeAffineMatrix').split(',')).astype(np.float)
    C = FisheyeAffineMat[0]
    D = FisheyeAffineMat[2]
    E = FisheyeAffineMat[1]
    F = FisheyeAffineMat[3]

    # get focal length and image dimensions
    FocalLength = float(meta.get_item('EXIF:FocalLength'))
    h, w = image.shape

    # get the two principal points
    pp = np.array(meta.get_item('XMP:PrincipalPoint').split(',')).astype(np.float)
    # values in pp and focallength are in [mm] and need to be rescaled to pixels
    FocalPlaneXResolution = float(meta.get_item('EXIF:FocalPlaneXResolution'))
    FocalPlaneYResolution = float(meta.get_item('EXIF:FocalPlaneYResolution'))
    cX = pp[0]*FocalPlaneXResolution
    cY = pp[1]*FocalPlaneYResolution
    fx = FocalLength * FocalPlaneXResolution
    fy = FocalLength * FocalPlaneYResolution

    # set up camera matrix
    cam_mat = np.zeros((3, 3))
    cam_mat[0, 0] = fx
    cam_mat[1, 1] = fy
    cam_mat[2, 2] = 1.0
    cam_mat[0, 2] = cX
    cam_mat[1, 2] = cY

    # create array with pixel coordinates
    x, y = np.meshgrid(range(w), range(h))
    P = np.array([x.flatten(order='F'), y.flatten(order='F'), np.ones(w * h)])

    # convert to normalized real world coordinates(z=1)
    p = np.linalg.solve(cam_mat, P) # p = inv(cam_mat) * P, same as Km\P in Matlab

    X = p[0]
    Y = p[1]

    X2 = X**2
    Y2 = Y**2

    sum = X2 + Y2

    r = np.sqrt(sum)

    theta = (2/math.pi) * np.arctan(r)

    row = p0 + p1 * theta + p2 * theta**2 + p3 * theta**3

    tmp = row / r

    Xh = X * tmp
    Yh = Y * tmp

    Xd = C * Xh + D * Yh + cX
    Yd = E * Xh + F * Yh + cY

    PDistorted = [Xd, Yd, np.ones(len(Xd))]

    # separate X and Y map from P
    XMap = np.reshape(PDistorted[0], (h, w), order='F')
    YMap = np.reshape(PDistorted[1], (h, w), order='F')

    # compute the undistorted 16 bit image
    undistortedImage = cv2.remap(image, XMap.astype(np.float32), YMap.astype(np.float32), cv2.INTER_LINEAR)
    return undistortedImage
