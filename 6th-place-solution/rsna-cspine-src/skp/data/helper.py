import numpy as np
import pydicom

from pydicom.pixel_data_handlers.util import apply_voi_lut



def apply_window(array, window):
    WL, WW = window
    WL, WW = float(WL), float(WW)
    lower, upper = WL - WW / 2, WL + WW / 2
    array = np.clip(array, lower, upper) 
    array = array - lower
    array = array / upper 
    return array 


def load_dicom(dcmfile, mode, convert_8bit=True, window=None, return_position=False,
               verbose=True):
    assert mode.lower() in ["xr", "ct", "mr"], f"{mode} is not a valid argument for `mode`"

    dicom = pydicom.dcmread(dcmfile)

    if return_position:
        assert hasattr(dicom, "ImagePositionPatient"), "DICOM metadata does not have ImagePositionPatient attribute"

    if mode == "xr": 
        # Apply the lookup table, if possible
        try:
            array = apply_voi_lut(dicom.pixel_array, dicom)
        except Exception as e:
            if verbose: print(e)
        # Rescale to [0, 1] using min and max values 
        array = array.astype("float32") 
        # Invert image, if needed
        if hasattr(dicom, "PhotometricInterpretation"):
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                array = np.amax(array) - array
        else:
            if verbose:
                print(f"{dcmfile} does not have attribute `PhotometricInterpretation`")
        array = array - np.min(array) 
        array = array / np.max(array)
        if convert_8bit:
            array = (array * 255.0).astype("uint8")

        array = np.expand_dims(array, axis=-1)

    elif mode == "ct": 
        # No need to apply LUT, we will assume values are HU
        # Rescale using intercept and slope
        M = float(dicom.RescaleIntercept)
        B = float(dicom.RescaleSlope)
        array = dicom.pixel_array.astype("float32")
        array = M*array + B
        # Apply window, if desired
        if isinstance(window, str):
            assert window == "raw_hu"
            array = np.expand_dims(array, axis=-1) 
        elif not isinstance(window, type(None)):
            array_list = []
            for each_window in window:
                array_list += [
                    np.expand_dims(apply_window(dicom, each_window), axis=-1)
                ]
            if convert_8bit:
                array_list = [(a * 255.0).astype("uint8") for a in array_list]
            # Each window is a dimension in the channel (last) axis
            array = np.concatenate(array_list, axis=-1) 
            # Sanity check
            assert array.shape[2] == len(window)
        else:
            raise Exception("You must provide window(s) or specify `raw_hu`")

    elif mode == "mr": 
        # Apply the lookup table, if possible
        try:
            array = apply_voi_lut(dicom.pixel_array, dicom)
        except Exception as e:
            if verbose: print(e)
        # Rescale to [0, 1] using min and max values 
        # Clip values to be within 2nd and 98th percentile
        array = array.astype("float32") 
        array = np.clip(array, np.percentile(array, 2), np.percentile(array, 98))
        array = array - np.min(array) 
        array = array / np.max(array)
        if convert_8bit:
            array = (array * 255.0).astype("uint8")

        array = np.expand_dims(array, axis=-1)

    if return_position: 
        return array, [float(i) for i in dicom.ImagePositionPatient]
        
    return array