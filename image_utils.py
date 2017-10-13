def tf_categorical_accuracy(pred, truth):
    """ Accuracy metric """
    return tf.reduce_mean(tf.cast(tf.equal(pred, truth), dtype=tf.float32))


def tf_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = tf.cast(tf.equal(pred, k), dtype=tf.float32)
    B = tf.cast(tf.equal(truth, k), dtype=tf.float32)
    return 2 * tf.reduce_sum(tf.multiply(A, B)) / (tf.reduce_sum(A) + tf.reduce_sum(B))


def data_augmenter(image, label, shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False):
    """ Online data augmentation """
    # Perform affine transformation on image and label, which are 4D tensor of shape (N, X, Y, C) and 3D tensor of shape (N, X, Y).
    image2 = np.zeros(image.shape, dtype=np.float32)
    label2 = np.zeros(label.shape, dtype=np.int32)
    for i in range(image.shape[0]):
        # Random affine transformation using normal distributions
        shift_val = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
        scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply affine transformation (rotation + scale + shift) to training images
        row, col = image.shape[1:3]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        for c in range(image.shape[3]):
            image2[i, :, :, c] = ndimage.interpolation.affine_transform(image[i, :, :, c], M[:, :2], M[:, 2], order=1)
        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_val

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, ::-1, :]
                label2[i] = label2[i, ::-1, :]
            else:
                image2[i] = image2[i, :, ::-1]
                label2[i] = label2[i, :, ::-1]
    return image2, label2

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