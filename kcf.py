import cv2
import numpy as np

"""
Python module that implements object tracking by Kernelized Correlation Filters (KCF)
Written by Rafael S. Formoso (rsformoso@gmail.com)
"""

def tracker(path, lamb=1e-4, roi=None, padding=2.5, kernel_sigma=10, 
            output_sigma_factor=0.1, interp_factor=0.075, cell_sz=1, features='hog'):
    """Kernelized Correlation Filter tracking.

    Keyword arguments:
    path -- The location of the video file to be read
    roi  -- An array of the form [x_pos, y_pos, x_sz, y_sz] which describes the ROI to be tracked
            If the input is None, prompts the user to select a bounding box
    padding -- The ROI is expanded by this percentage for purposes of tracking
    sigma_factor -- The variance of the training output (i.e. how heavily each shift is penalized)
    interp_factor -- The rate of adaptation of the tracker
    cell_sz -- The number of pixels per cell (used for generating HOG descriptors)
    features -- The type of feature to be used in the model
    """

    cap = cv2.VideoCapture(path)

    # If no ROI was specified in the arguments, asks users to select one from the first frame
    if roi is None:
        _, frame = cap.read()
        roi = cv2.selectROI(frame)
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(path)

    roi = np.array(roi)
    orig_roi = roi.copy()

    # Pads the ROI and moves the bounding box accordingly
    roi_center = np.array([roi[0] + roi[2] * 0.5, roi[1] + roi[3] * 0.5])
    roi[2] = np.floor(roi[2] * (1 + padding))
    roi[3] = np.floor(roi[3] * (1 + padding))
    roi[0] = np.floor(roi_center[0] - roi[2] / 2)
    roi[1] = np.floor(roi_center[1] - roi[3] / 2)

    # Modify this when other features have been applied
    # This is the size of the feature matrix, in the form [x_sz, y_sz]
    window_sz = roi[2:4]

    # Generates the output matrix to be used for training. "Penalizes" every shift in the image
    output_sigma = np.sqrt(np.prod(roi[2:4])) * output_sigma_factor
    y_f = np.fft.fft2(gen_label(output_sigma, window_sz))

    num_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        num_frame += 1
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = frame.reshape(frame.shape[0], frame.shape[1], 1)

        if num_frame > 1:
            x = get_subwindow(frame, roi)

            x_f = np.fft.fft2(x, axes=(0, 1))
            kz_f = gaussian_correlation(x_f, xhat_f, kernel_sigma)

            # Searches for the shift with the greatest response to the model
            # Note that shift is of the form [y_pos, x_pos]
            resp = np.real(np.fft.ifft2(kz_f * ahat_f))
            shift = np.unravel_index(resp.argmax(), resp.shape)
            shift = np.array(shift) + 1

            # If the shift is higher than halfway, then it is interpreted as being a shift
            # in the opposite direction (i.e., right by default, but becomes left if too high)
            if shift[0] > x_f.shape[0] / 2:
                shift[0] -= x_f.shape[0]
            if shift[1] > x_f.shape[1] / 2:
                shift[1] -= x_f.shape[1]

            roi[0] += shift[1] - 1
            roi[1] += shift[0] - 1

            # Moves the original ROI similarly for display purposes
            orig_roi[0] += shift[1] - 1
            orig_roi[1] += shift[0] - 1

        # Trains the model given the current ROI
        x = get_subwindow(frame, roi)

        x_f = np.fft.fft2(x, axes=(0,1))
        k_f = gaussian_correlation(x_f, x_f, kernel_sigma)
        a_f = y_f / (k_f + lamb)

        # Stores the model's parameters. x_hat is kept for the calculation of kernel correlations
        if num_frame == 1:
            ahat_f = a_f.copy()
            xhat_f = x_f.copy()
        else:
            ahat_f = (1 - interp_factor) * ahat_f + interp_factor * a_f
            xhat_f = (1 - interp_factor) * xhat_f + interp_factor * x_f

        #cv2.imshow('x', get_subwindow(frame, orig_roi))
        cv2.imshow('x', cv2.rectangle(frame, tuple(orig_roi[0:2]), tuple(orig_roi[0:2] + orig_roi[2:4]), (0, 0, 255), 2))
        #cv2.imshow('x', cv2.rectangle(frame, tuple(roi[0:2]), tuple(roi[0:2] + roi[2:4]), (0, 0, 255), 2))
        if cv2.waitKey(1) == 27:
            break


def get_subwindow(frame, roi):
    """Returns an array representing the subwindow of a frame, as defined by the ROI.

    Keyword arguments:
    frame -- The array representing the image. By default, must be of the form [y, x, ch]
    roi -- The array representing the ROI to be extracted. By default, is of the form
           [x_pos, y_pos, x_sz, y_sz]
    """        
    if roi[0] < 0:
        roi[0] = 0
    if roi[1] < 0:
        roi[1] = 0

    if roi[0] + roi[2] > frame.shape[1]:
        roi[0] = frame.shape[1] - roi[2]

    if roi[1] + roi[3] > frame.shape[0]:
        roi[1] = frame.shape[0] - roi[3]

    return frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]


def gen_label(sigma, sz):
    """Generates a matrix representing the penalty for shifts in the image

    Keyword arguments:
    sigma -- The standard deviation of the Gaussian model
    sz -- An array of the form [x_sz, y_sz] representing the size of the feature array
    """
    sz = np.array(sz).astype(int)
    rs, cs = np.meshgrid(range(1, sz[0]+1) - np.floor(sz[0]/2), range(1, sz[1]+1) - np.floor(sz[1]/2))
    labels = np.exp(-0.5 / sigma ** 2 * (rs ** 2 + cs ** 2))

    # The [::-1] reverses the sz array, since it is of the form [x_sz, y_sz] by default
    labels = np.roll(labels, np.floor(sz[::-1] / 2).astype(int), axis=(0,1))

    return labels


def gaussian_correlation(x_f, y_f, sigma):
    """Calculates the Gaussian correlation between two images in the Fourier domain.

    Keyword arguments:
    x_f -- The representation of x in the Fourier domain
    y_f -- The representation of y in the Fourier domain
    sigma -- The variance to be used in the calculation
    """
    N = x_f.shape[0] * x_f.shape[1]
    xx = np.real(x_f.flatten().conj().dot(x_f.flatten()) / N)
    yy = np.real(y_f.flatten().conj().dot(y_f.flatten()) / N)

    xy_f = np.multiply(x_f, y_f.conj())
    xy = np.sum(np.real(np.fft.ifft2(xy_f, axes=(0,1))), 2)

    k_f = np.fft.fft2(np.exp(-1 / (sigma ** 2) * ((xx + yy - 2 * xy) / x_f.size).clip(min=0)))

    return k_f


if __name__ == '__main__':
    #tracker('movie.mp4', roi=[555, 190, 62, 140])

