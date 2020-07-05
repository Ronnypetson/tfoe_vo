import numpy as np
import cv2


def intersecc_(points, dirs):
    """
    :param points: (N, 2) array of points on the lines
    :param dirs: (N, 2) array of unit direction vectors
    :returns: (2,) array of intersection point
    """
    dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
    points_mat = points[:, :, np.newaxis]
    I = np.eye(2)
    return np.linalg.lstsq(
        (I - dirs_mat).sum(axis=0),
        ((I - dirs_mat) @ points_mat).sum(axis=0),
        rcond=None)[0]


def intersecc(lines):
    """
    :param lines: n,3
    :returns: (2,) array of intersection point
    """
    a = lines[:, :-1]
    b = -lines[:, -1]
    p = np.linalg.lstsq(a, b, rcond=None)[0]
    return p


def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        #img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        #img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2
