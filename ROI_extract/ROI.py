import numpy as np
from scipy.spatial import Delaunay
import cv2
import math

def ROI(shape,region):
    if region=='nose':
        w = 0.7 * (shape.part(42).x - shape.part(39).x)
        delta_x = w / 2
        x0 = shape.part(27).x - 1.25*delta_x
        y0 = shape.part(27).y

        x1 = shape.part(30).x + 1.25*delta_x
        y1 = shape.part(30).y

    if region == 'leftcheek':
        h = abs((shape.part(33).y - shape.part(27).y))
        w = abs((shape.part(41).x - shape.part(39).x))
        delta_x = w / 2
        delta_y = h / 2
        x0 = shape.part(46).x - 1.5*delta_x
        y0 = shape.part(46).y + delta_y

        x1 = shape.part(46).x + 1.5*delta_x
        y1 = shape.part(46).y + h

    if region == 'rightcheek':
        h = abs((shape.part(33).y - shape.part(27).y))
        w = abs((shape.part(41).x - shape.part(39).x))
        delta_x = w / 2
        delta_y = h / 2
        x0 = shape.part(41).x - 1.5*delta_x
        y0 = shape.part(41).y + delta_y

        x1 = shape.part(41).x + 1.5*delta_x
        y1 = shape.part(41).y + h

    return int(x0),int(y0),int(x1),int(y1)
