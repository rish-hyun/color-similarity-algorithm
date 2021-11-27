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
    try:
        img = {}
        img['BYTES'] = BytesIO(requests.get(url).content)
        img['PIL'] = Image.open(img['BYTES'])
        img['CV2'] = cv2.cvtColor(np.array(img['PIL']), cv2.COLOR_RGB2BGR)
    except:
        img = None
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
            'COLORTHIEF': self.__COLORTHIEF__(self.img['BYTES']),
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

        # ------- cruv_method -------
        # cruv_col = self.__COLORTHIEF__(self.img['BYTES'])
        # color_keys.append(label('CRUV_METHOD', width=color_key_width))
        # dom_color_map.append(visualize(cruv_col))
        # color_map.append(visualize([[1, self.cruv_method(cruv_col[0], color_list)]]))
        # ------- cruv_method -------

        img = image_resize(self.img['CV2'], height=HEIGHT*(len(color_map)-1))
        img = [label('Image', width=img.shape[1]), img]

        comp_map = np.hstack([np.vstack(color_keys),
                              np.vstack(dom_color_map),
                              np.vstack(img),
                              np.vstack(color_map)])

        cv2.imwrite(file_name, comp_map)
        return

    # --------------------------------------------------------------------------

    @timer
    def cruv_method(self, color, color_list, file_name):
        from collections import Counter

        id = -1
        rejected = False
        if not rejected:

            id_assigned = False

            #print("OK, GOing to classify it")
            r, g, b = color
            l_, a_, b_ = self.rgb2lab(color)
            #print(f'L:{l_}, A:{a_}, B:{b_}')

            if l_ < 50:
                if abs(a_) < 3 and abs(b_) < 9:
                    if l_ < 20:
                        color_name = 'BLACK'
                        id_assigned = True
                    else:
                        color_name = 'GRAY'
                        id_assigned = True

            if (abs(a_) < 3 and abs(b_) < 6):
                if l_ > 50 and l_ < 75:
                    color_name = 'GRAY'
                    id_assigned = True
                elif l_ >= 75:
                    color_name = 'WHITE'
                    id_assigned = True
                elif l_ <= 35:
                    color_name = 'BLACK'
                    id_assigned = True

            if not id_assigned:

                a = self.rgb2lab([r, g, b])

                score_list = []
                color_matrix_non_bgw = {
                    'DARK_RED': [(78, 7, 7), (66, 12, 9), (97, 12, 4), (104, 12, 7), (96, 16, 11), (113, 12, 4), (144, 6, 3), (94, 25, 22), (84, 30, 27), (144, 13, 9)],
                    'LIGHT_RED': [(122, 23, 18), (153, 15, 2), (155, 16, 3), (185, 14, 10), (126, 40, 17), (169, 27, 13), (210, 20, 4), (227, 36, 43), (208, 49, 45), (188, 84, 75)],
                    'DARK_ORANGE': [(137, 49, 1), (122, 56, 3), (128, 64, 11), (141, 64, 4), (178, 86, 13), (190, 85, 4), (204, 88, 1), (201, 91, 12), (221, 87, 28), (209, 96, 2)],
                    'LIGHT_ORANGE': [(181, 103, 39), (214, 114, 41), (252, 106, 3), (237, 112, 20), (237, 113, 23), (237, 130, 14), (250, 129, 40), (236, 151, 6), (253, 161, 114), (252, 174, 30)],
                    'DARK_YELLOW': [(189, 165, 93), (200, 169, 81), (214, 184, 90), (216, 184, 99), (201, 187, 142), (227, 183, 120), (218, 193, 124), (227, 197, 101), (231, 194, 125), (223, 201, 138)],
                    'LIGHT_YELLOW': [(220, 215, 160), (230, 219, 172), (238, 220, 154), (249, 224, 118), (250, 226, 156), (251, 231, 144), (237, 232, 186), (253, 233, 146), (243, 234, 175), (253, 239, 178)],
                    'DARK_GREEN': [(35, 79, 30), (53, 74, 33), (58, 83, 17), (50, 97, 45), (70, 109, 29), (2, 138, 15), (89, 125, 53), (96, 125, 59), (3, 172, 19), (114, 140, 105)],
                    'LIGHT_GREEN': [(3, 192, 74), (60, 176, 67), (116, 183, 46), (93, 187, 99), (152, 191, 100), (61, 237, 151), (178, 211, 194), (153, 237, 195), (174, 243, 89), (176, 252, 56)],
                    'DARK_BLUE': [(10, 17, 114), (5, 16, 148), (21, 30, 61), (36, 21, 113), (2, 45, 54), (40, 30, 93), (21, 32, 166), (19, 56, 190), (40, 50, 194), (44, 62, 76)],
                    'LIGHT_BLUE': [(31, 69, 110), (1, 96, 100), (57, 68, 188), (89, 120, 142), (4, 146, 194), (117, 124, 136), (72, 170, 173), (82, 178, 191), (99, 197, 218), (130, 238, 253)],
                    'DARK_PURPLE': [(44, 4, 28), (41, 9, 22), (76, 1, 33), (49, 20, 50), (103, 3, 47), (99, 4, 54), (77, 15, 40), (113, 1, 147), (96, 26, 53), (161, 4, 90)],
                    'LIGHT_PURPLE': [(102, 48, 70), (163, 44, 196), (122, 73, 136), (164, 94, 229), (152, 103, 197), (182, 95, 207), (175, 105, 239), (158, 123, 181), (190, 147, 212), (227, 159, 246)],
                    'DARK_PINK': [(225, 21, 132), (255, 22, 148), (158, 66, 68), (252, 76, 78), (252, 70, 170), (242, 82, 120), (253, 93, 168), (242, 107, 138), (254, 125, 106), (254, 127, 156)],
                    'LIGHT_PINK': [(250, 134, 196), (252, 148, 131), (252, 148, 175), (246, 153, 205), (247, 154, 192), (253, 164, 186), (253, 171, 159), (242, 184, 198), (252, 186, 203), (254, 197, 229)],
                    'DARK_BROWN': [(35, 23, 9), (46, 21, 3), (55, 29, 16), (59, 30, 8), (53, 35, 21), (72, 31, 1), (54, 37, 17), (60, 40, 13), (67, 38, 22), (74, 37, 17)],
                    'LIGHT_BROWN': [(83, 41, 21), (63, 48, 29), (94, 44, 4), (101, 42, 14), (75, 55, 28), (74, 55, 40), (101, 53, 15), (128, 71, 28), (121, 92, 52), (154, 123, 79)]
                }
                for key, val in color_matrix_non_bgw.items():
                    for color_2 in val:
                        b = self.rgb2lab([color_2[0], color_2[1], color_2[2]])
                        score_list.append([key,
                                        [color_2[0], color_2[1], color_2[2]],
                                        delta_E(a, b, 'CIE 1976'),
                                        delta_E(a, b, 'CIE 2000')])

                score_rank_1976 = sorted(score_list, key=lambda x: x[2])[0:1]
                score_rank_2000 = sorted(score_list, key=lambda x: x[3])[0:1]

                color_name = Counter(val[0] for val in score_rank_1976).most_common()[0][0]

                if color_name != 'BROWN':
                    color_name = Counter(val[0] for val in score_rank_2000).most_common()[0][0]
                    # print('delta_e_cie2000')
                    # print(score_rank_2000)
                    #print(Counter(val[0] for val in score_rank_2000).most_common())
                    #visualize([(1, color), (1, score_rank_2000[0][1])])
                else:
                    pass
                    # print('score_rank_1976')
                    # print(score_rank_1976)
                    #print(Counter(val[0] for val in score_rank_1976).most_common())
                    #visualize([(1, color), (1, score_rank_1976[0][1])])

            #print(f'{color} Similar to {color_name}')
            #visualize([(1, color), (1, [[val['r'], val['g'], val['b']]for val in color_ids if val['name'].upper() == color_name][0])])

            id = [val['name'] for val in color_list if val['name'].upper() == color_name][0]

        cruv_color = [[c['r'], c['g'], c['b']] for c in color_list if c['name'] == color_name][0]

        color_key_width = 200
        color_keys = [label(text='Sample Color', width=color_key_width)]
        color_keys.append(label('CIE 1976', width=color_key_width))
        color_keys.append(label('CIE 2000', width=color_key_width))
        color_keys.append(label('CRUV METHOD', width=color_key_width))

        a = self.rgb2lab(color)
        def method(method_name):
            score = []
            for col in color_list:
                b = self.rgb2lab([col['r'], col['g'], col['b']])
                score.append(([col['r'], col['g'], col['b']], col['name'], delta_E(a, b, method_name)))
            return sorted(score, key=lambda x: x[2])[0][0]

        color_map = [visualize([[1,color]])]
        color_map.append(visualize([[1,method('CIE 1976')]]))
        color_map.append(visualize([[1,method('CIE 2000')]]))
        color_map.append(visualize([[1,cruv_color]]))

        comp_map = np.hstack([np.vstack(color_keys),
                              np.vstack(color_map)])

        cv2.imwrite(file_name, comp_map)

    # --------------------------------------------------------------------------
