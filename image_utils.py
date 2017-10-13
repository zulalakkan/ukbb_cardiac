def crop_image(image, cx, cy, size):
    # Crop a 3D image using a bounding box centred at (cx, cy) and with specified size
    X, Y = image.shape[:2]
    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    crop = image[x1_: x2_, y1_: y2_]
    if crop.ndim == 3:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)), 'constant')
    elif crop.ndim == 4:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)), 'constant')
    else:
        print('Error: crop.ndim = {0}. Unsupported.'.format(crop.ndim))
        exit(0)
    return crop


def scale_intensity(image, thres=(1.0, 99.0)):
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2