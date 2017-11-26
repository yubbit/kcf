Kernelized Correlation Filters in Python
Written by Rafael S. Formoso (rsformoso@gmail.com)

---

Python module that implements object tracking by Kernelized Correlation Filters (KCF)

This aims to build an object tracking pipeline around the use of a KCF-based tracker. At present
this is a direct translation of João Henriques' original MATLAB code.

When examining this code, note that when used to select an ROI, the selectROI function returns a 
value of the form [x_pos, y_pos, x_sz, y_sz], which does not follow the standard [y, x] indexing 
of numpy. In addition, numpy's FFT implementation works on the last two indices, in contrast with
MATLAB, which uses the first two indices, interpreting each image array as [x, y, ch]

Usage: Call the tracker() function from the kcf module, specifying the path to the video file.
Other parameters can be applied as seen in the module itself.
