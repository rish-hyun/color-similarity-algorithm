import cv2
import time
import requests
import functools
import numpy as np
from PIL import Image
from io import BytesIO
from colorthief import ColorThief
from sklearn.cluster import KMeans
from extcolors import extract_from_image
from colour import delta_E, DELTA_E_METHODS

WIDTH = 600
HEIGHT = 100


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f">>> | {repr(func.__name__)} : {round(run_time, 3)} secs | <<<")
        return value
    return wrapper


def url_to_image(url):
    img = {}
    img['BYTES'] = BytesIO(requests.get(url).content)
    img['PIL'] = Image.open(img['BYTES'])
    img['CV2'] = cv2.cvtColor(np.array(img['PIL']), cv2.COLOR_RGB2BGR)
    return img


def visualize(colors, size=(HEIGHT, WIDTH)):
    height, width = size
    rect = np.zeros((height, width, 3), dtype=np.uint8)
    start = 0

    if len(colors[0]) == 3:
        colors = [[1, color] for color in colors]

    denominator = sum([per[0] for per in colors])
    for (percent, color) in colors:
        end = start + ((percent / denominator) * width)
        color = [int(col) for col in color]
        cv2.rectangle(rect, (int(start), 0), (int(end), height), color, -1)
        start = end

    return cv2.cvtColor(rect, cv2.COLOR_RGB2BGR)


def label(text=None, width=WIDTH, height=HEIGHT):
    thickness = 1
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX

    size = (height, width)
    img = np.zeros((*size, 3), dtype="uint8")

    if text is not None:
        x_offset = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        x, y = (size[1]-x_offset)//2, (size[0])//2
        img = cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    return img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


class Color:

    def __init__(self, url) -> None:
        self.img = url_to_image(url)

    # --------------------------------------------------------------------------

    def __EXTCOLORS__(self, img):
        ext_colors, pixel_count = extract_from_image(img)
        colors = [[(percent/pixel_count)*100, color]
                  for (color, percent) in ext_colors]
        return [colors[0][-1]]

    # --------------------------------------------------------------------------

    def __KMEANS__(self, img, k_clusters):
        reshape_img = img.reshape((img.shape[0] * img.shape[1], 3))
        cluster = KMeans(k_clusters, init='k-means++').fit(reshape_img)
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins=labels)
        hist = hist.astype("float")
        hist /= hist.sum()
        kmeans_colors = [(percent, color[::-1])
                         for (percent, color) in zip(hist, cluster.cluster_centers_)]
        return [sorted(kmeans_colors, reverse=True)[0][-1]]

    # --------------------------------------------------------------------------

    def __RESIZE__(self, img):
        return [tuple(cv2.resize(img, (1, 1))[0][0][::-1])]

    # --------------------------------------------------------------------------

    def __AVERAGE__(self, img):
        return [tuple(img.mean(axis=0).mean(axis=0)[::-1])]

    # --------------------------------------------------------------------------

    def __COLORTHIEF__(self, img):
        return [ColorThief(img).get_color(quality=1)]

    # --------------------------------------------------------------------------

    def rgb2lab(self, inputColor):

        num = 0
        RGB = [0, 0, 0]

        for value in inputColor:
            value = float(value) / 255

            if value > 0.04045:
                value = ((value + 0.055) / 1.055) ** 2.4
            else:
                value = value / 12.92

            RGB[num] = value * 100
            num = num + 1

        XYZ = [0, 0, 0, ]

        X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
        Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
        Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
        XYZ[0] = round(X, 4)
        XYZ[1] = round(Y, 4)
        XYZ[2] = round(Z, 4)

        # ref_X =  95.047   Observer= 2°, Illuminant= D65
        XYZ[0] = float(XYZ[0]) / 95.047
        XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
        XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

        num = 0
        for value in XYZ:

            if value > 0.008856:
                value = value ** (0.3333333333333333)
            else:
                value = (7.787 * value) + (16 / 116)

            XYZ[num] = value
            num = num + 1

        Lab = [0, 0, 0]

        L = (116 * XYZ[1]) - 16
        a = 500 * (XYZ[0] - XYZ[1])
        b = 200 * (XYZ[1] - XYZ[2])

        Lab[0] = round(L, 4)
        Lab[1] = round(a, 4)
        Lab[2] = round(b, 4)

        return Lab

    # --------------------------------------------------------------------------

    def __extract_colors__(self):
        color_list = {
            'EXTCOLORS': self.__EXTCOLORS__(self.img['PIL']),
            'KMEANS': self.__KMEANS__(self.img['CV2'], k_clusters=5),
            'RESIZE': self.__RESIZE__(self.img['CV2']),
            'AVERAGE': self.__AVERAGE__(self.img['CV2']),
            'COLORTHIEF': self.__COLORTHIEF__(self.img['BYTES'])
        }
        return color_list

    # --------------------------------------------------------------------------

    def __match_color__(self, color, color_list):
        a = self.rgb2lab(color)
        delta_scores = {}
        for method in DELTA_E_METHODS:
            score = []
            for col in color_list:
                b = self.rgb2lab([col['r'], col['g'], col['b']])
                score.append((col['name'], delta_E(a, b, method)))
            delta_scores[method] = sorted(score, key=lambda x: x[1])[0]

        score = list(zip(*delta_scores.values()))[0]
        score = {i: score.count(i)/len(DELTA_E_METHODS) for i in score}
        return score

    # --------------------------------------------------------------------------

    @timer
    def get_match(self, color_list, file_name):
        color_key_width = 200

        color_keys = [label(text='X', width=color_key_width)]
        dom_color_map = [label('Extracted Color')]
        color_map = [label('Assigned Color')]

        for key, col in self.__extract_colors__().items():
            color_keys.append(label(key, width=color_key_width))
            dom_color_map.append(visualize(col))
            res = self.__match_color__(col[0], color_list)

            colors = []
            for col_name, perc in res.items():
                color = [[c['r'], c['g'], c['b']]
                         for c in color_list if c['name'] == col_name][0]
                colors.append([perc, color])
            color_map.append(visualize(colors))

        img = image_resize(self.img['CV2'], height=HEIGHT*(len(color_map)-1))
        img = [label('Image', width=img.shape[1]), img]

        comp_map = np.hstack([np.vstack(color_keys),
                              np.vstack(dom_color_map),
                              np.vstack(img),
                              np.vstack(color_map)])

        cv2.imwrite(file_name, comp_map)
        return

    # --------------------------------------------------------------------------
