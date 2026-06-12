# Standard library imports
import os
from typing import Any, Dict, Optional, Tuple, Union

# Third-party library imports
import cv2
import numpy as np
import pandas as pd
from scipy.stats import linregress, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import find_contours, perimeter
from skimage.morphology import convex_hull_image, skeletonize

# Type hints


def summary_by(df: pd.DataFrame, summarise_by: str, variable: str) -> pd.DataFrame:
    """
    Summarize a variable in a DataFrame by a specified grouping variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        summarise_by (str): The column name to group the data by.
        variable (str): The column name of the variable to summarize.

    Returns:
        pd.DataFrame: A DataFrame containing summary statistics for the specified variable
        grouped by the specified grouping variable. The summary statistics include mean,
        max, min, standard deviation, 95th percentile, 5th percentile, and count.
    """
    result = df.groupby(summarise_by)[variable].agg(
        [
            ("mean", "mean"),
            ("max", "max"),
            ("min", "min"),
            ("std", "std"),
            ("quantile_95", lambda x: x.quantile(0.95)),
            ("quantile_05", lambda x: x.quantile(0.05)),
            ("n", "count"),
        ]
    )

    return result


def read_csv_files(
    folder: str, idcol: str, pattern: Optional[str] = None, **kwargs
) -> Optional[pd.DataFrame]:
    """
    Read all CSV files in a folder and combine them into a single DataFrame.

    Args:
        folder (str): Path to the folder containing the CSV files.
        idcol (str): Name of the column to add to the DataFrame, containing the file name.
        pattern (str, optional): Pattern to filter the CSV files. Only files containing the pattern will be read.
        **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
        pd.DataFrame: Combined DataFrame from all the CSV files.
        None: If no CSV files are found in the specified folder.
    """
    # List csv files and filter by pattern
    csv_files = [file for file in os.listdir(folder) if file.endswith(".csv")]
    if pattern:
        csv_files = [file for file in csv_files if pattern in file]

    # Read all CSV files and combine
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, **kwargs)
        df[idcol] = file
        df_list.append(df)

    # Concat
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        print("No CSV files found in the specified folder.")
        return None


def cluster_binder(
    dictionary: Dict[Any, list], dataframe: pd.DataFrame, target_column: str
) -> pd.Series:
    """
    Bind cluster labels to values in a DataFrame column based on a dictionary mapping.

    Args:
        dictionary (Dict[Any, list]): Dictionary mapping cluster labels to lists of values.
        dataframe (pd.DataFrame): DataFrame containing the target column.
        target_column (str): Name of the column in the DataFrame to bind cluster labels to.

    Returns:
        pd.Series: Series containing the bound cluster labels for each value in the target column.
    """

    def find_key(value: Any) -> Optional[Any]:
        """
        Find the key in the dictionary that contains the given value.

        Args:
            value (Any): Value to search for in the dictionary.

        Returns:
            Optional[Any]: Key containing the value, or None if not found.
        """
        for key, item_list in dictionary.items():
            if value in item_list:
                return key
        return None

    new_col = dataframe[target_column].apply(find_key)

    return new_col


def create_array_from_coords(
    df: pd.DataFrame,
    img_shape: Tuple[int, int],
    value_column: str,
    coords_column: str = "coords_list",
    batch_column: Union[int, str] = 0,
    to_one_hot: bool = False,
) -> np.ndarray:
    """
    Create an array from coordinates and corresponding values.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing all the data.
        img_shape (tuple): Shape of the output array in the format (height, width).
        value_column (str): Name of the column in df containing the values for each coordinate.
        coords_column (str): Name of the column in df containing the coordinate tuples (x, y). Default is 'coords_list'.
        batch_column (Union[int, str]): If int, all coordinates are assigned to this batch. If str, name of the column in df containing the batch indices for each coordinate. Default is 0.
        to_one_hot (bool): Whether to convert the array to one-hot encoding. Default is False.

    Returns:
        np.ndarray: Array with values assigned to the specified coordinates.
    """

    # Prepare data
    value_column = df[value_column]
    coords_column = df[coords_column]

    if batch_column == 0:
        batch_column = [0] * len(coords_column)
    else:
        batch_column = df[batch_column]

    num_batches = max(batch_column) + 1
    array_stack = np.zeros((num_batches,) + img_shape)
    num_classes = len(np.unique(value_column))

    # Iterate over coords
    for batch, coords, value in zip(batch_column, coords_column, value_column):
        x_coords, y_coords = zip(*coords)
        array_stack[batch, x_coords, y_coords] = value

    # Return one-hot-encoded data if requested
    if to_one_hot:
        value_column = value_column.astype(int)
        array_stack = one_hot_encode(array_stack, num_classes)

    # Adjust dtype for memory efficiency
    if num_classes < 256 or to_one_hot:
        array_stack = array_stack.astype(np.uint8)
    elif num_classes < 65536:
        array_stack = array_stack.astype(np.uint16)
    else:
        array_stack = array_stack.astype(np.uint32)

    return array_stack


def one_hot_encode(image_stack, num_classes):
    """
    Transform multiclass image stack to one-hot encoding. Classes must be integers and channel dimension
    will be used to encode the different classes.

    Args:
        image_stack (numpy.ndarray): Image stack with integer values representing classes.
        num_classes (int): Number of classes in the image stack.

    Returns:
        numpy.ndarray: One-hot encoded image stack with the channel dimension moved to the second position.
    """
    identity = np.eye(num_classes)
    one_hot_encoded = identity[image_stack]
    one_hot_encoded = np.moveaxis(one_hot_encoded, -1, 1)

    return one_hot_encoded


def create_array_from_coords_old(
    df: pd.DataFrame,
    x_dim: int,
    y_dim: int,
    z_dim: int,
    z_col: str,
    coord_col: str = "coords_list",
) -> np.ndarray:
    """
    Create a 3D numpy array from coordinates stored in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinates and corresponding frame (z) values.
        x_dim (int): Dimension of the array along the x-axis.
        y_dim (int): Dimension of the array along the y-axis.
        z_dim (int): Dimension of the array along the z-axis.
        z_col (str): Name of the column in the DataFrame containing the frame (z) values.
        coord_col (str, optional): Name of the column in the DataFrame containing the coordinate tuples (x, y). Default is 'coords_list'.

    Returns:
        np.ndarray: 3D numpy array with values set to 1 at the specified coordinates.
    """
    # Create a black numpy array of dimensions x, y, z
    array = np.zeros((z_dim, x_dim, y_dim), dtype=float)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Get the x, y coordinates and the corresponding frame (z)
        coords = row[coord_col]
        frame = row[z_col]
        # Update the pixel values in the array
        for coord in coords:
            x, y = coord
            array[frame, x, y] = 1

    return array


# Image analysis functions
def border_complexity(regionmask, intensity):
    """
    Calculate the border complexity of a region by comparing the perimeter of the region
    to the perimeter of its convex hull.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values (unused in this function).

    Returns:
    float: The border complexity value, defined as the ratio of the region's perimeter
           to the perimeter of its convex hull.
    """
    try:
        # Find contours
        contours = find_contours(regionmask, 0.5)
        if len(contours) == 0:
            return 0.0

        # Get perimeter length and convex hull length
        region_perimeter = perimeter(regionmask)
        hull = convex_hull_image(regionmask)
        hull_perimeter = perimeter(hull)

        # Calculate the border complexity
        if hull_perimeter != 0:
            border_complexity = region_perimeter / hull_perimeter
        else:
            border_complexity = 1.0
    except Exception:
        border_complexity = 0.0

    return border_complexity


## Auxiliary functions
def rod_shape_coef(regionmask, intensity):
    """
    Skeletonize the region mask, perform a fast linear regression, and return the R-squared value.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values (unused in this function).

    Returns:
    float: The R-squared value of the linear regression on the skeletonized region.
    """
    # Get skeletopn coords
    skeleton = skeletonize(regionmask)
    y, x = np.where(skeleton)

    if len(x) < 2:
        return 0.0

    if np.all(x == x[0]):
        return 1.0

    # Calculate the R-squared value
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2
    except ValueError:
        r_squared = 0

    return r_squared


def coords_centroid(coords):
    """Return centroid coordinates as a Series indexed by stack dimension."""
    centroid = np.mean(coords, axis=0)
    return pd.Series(centroid, index=["slice", "y", "x"])


def quartiles(regionmask, intensity):
    """
    Calculate the quartiles of the given intensity values within the specified region mask.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values.

    Returns:
    ndarray: An array containing the 25th, 50th, and 75th percentiles of the intensity values within the region mask.
    """
    return np.percentile(intensity[regionmask], q=(25, 50, 75))


def roi_skewness(regionmask, intensity):
    """
    Calculate the skewness of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (numpy.ndarray): A binary mask defining the ROI.
    intensity (numpy.ndarray): The intensity image.

    Returns:
    float: The skewness of pixel intensities within the ROI.
    """
    roi_intensities = intensity[regionmask]

    try:
        # Check if there are enough unique values in roi_intensities
        unique_values = np.unique(roi_intensities)
        if len(unique_values) < 10:
            return 0

        return skew(roi_intensities, bias=False)
    except Exception:
        return 0


def roi_std_dev(regionmask, intensity):
    """
    Calculate the standard deviation of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (numpy.ndarray): A binary mask defining the ROI.
    intensity (numpy.ndarray): The intensity image.

    Returns:
    float: The standard deviation of pixel intensities within the ROI.
    """
    roi_intensities = intensity[regionmask]
    return np.std(roi_intensities)


def laplacian(image):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian.
    """
    image = np.float32(image)

    # Check the data type of the image
    if image.dtype == np.float32:
        ddepth = cv2.CV_32F
    elif image.dtype == np.float64:
        ddepth = cv2.CV_64F
    else:
        raise ValueError(f"Unsupported image data type: {image.dtype}")

    return cv2.Laplacian(image, ddepth)


def variance_filter(image, kernel_size):
    """
    Compute the local variance of an image using a square averaging kernel.

    Args:
        image (np.ndarray): Input image (will be coerced to float32).
        kernel_size (int): Size of the square window applied with cv2.blur.

    Returns:
        np.ndarray: Variance map with the same shape as the input image.
    """
    # Convert the image to float32
    image = np.float32(image)

    # Calculate the mean of the image
    mean = cv2.blur(image, (kernel_size, kernel_size))
    mean_sqr = cv2.blur(np.square(image), (kernel_size, kernel_size))

    # Calculate the variance
    variance = mean_sqr - np.square(mean)

    return variance


def dilated_measures(regionmask, intensity, structure=np.ones((5, 5)), iterations=1):
    """
    Calculate the standard deviation of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (numpy.ndarray): A binary mask defining the ROI.
    intensity (numpy.ndarray): The intensity image.
    structure (numpy.ndarray): The structuring element used for dilation.
    iterations (int): The number of times dilation is applied.

    Returns:
    float: The standard deviation of pixel intensities within the ROI.
    """
    # Ensure regionmask is an 8-bit, single-channel image
    regionmask = regionmask.astype(np.uint8)

    # Dilate the regionmask
    dilated_regionmask = cv2.dilate(regionmask, structure, iterations=iterations)

    # Get the intensities within the dilated ROI
    roi_intensities = intensity[dilated_regionmask > 0]
    std_px = np.std(roi_intensities)
    mean_px = np.mean(roi_intensities)
    min_px = np.min(roi_intensities)
    max_px = np.max(roi_intensities)
    pixel_area = np.sum(dilated_regionmask > 0)

    return (std_px, mean_px, min_px, max_px, pixel_area)


def roi_glcm_features(regionmask: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """GLCM texture features for a single ROI.

    Computes a Grey-Level Co-occurrence Matrix over 4 angles (0°, 45°, 90°, 135°)
    at distance=1 with 64 grey levels. Returns the mean over all angles for each
    of four Haralick properties.

    Returns a 4-element array: [contrast, homogeneity, energy, correlation].
    When added via extra_properties to regionprops_table, the output columns are
    roi_glcm_features-0 … roi_glcm_features-3; rename them after extraction.
    """
    rows, cols = np.where(regionmask)
    if len(rows) < 9:
        return np.zeros(4, dtype=float)

    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    roi_patch = intensity[r0 : r1 + 1, c0 : c1 + 1].astype(float)
    mask_patch = regionmask[r0 : r1 + 1, c0 : c1 + 1]

    # Normalize to 0–63 within the ROI; background pixels stay 0
    roi_vals = roi_patch[mask_patch]
    v_min, v_max = roi_vals.min(), roi_vals.max()
    if v_max > v_min:
        scaled = ((roi_patch - v_min) / (v_max - v_min) * 63).astype(np.uint8)
    else:
        scaled = np.zeros_like(roi_patch, dtype=np.uint8)
    scaled[~mask_patch] = 0

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    try:
        glcm = graycomatrix(scaled, distances=[1], angles=angles, levels=64,
                            symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, "contrast").mean())
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())
        energy = float(graycoprops(glcm, "energy").mean())
        correlation = float(graycoprops(glcm, "correlation").mean())
    except Exception:
        return np.zeros(4, dtype=float)

    return np.array([contrast, homogeneity, energy, correlation])


def roi_radial_profile(regionmask: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Mean FL intensity in 5 equal-width radial bins from centroid to edge.

    Bins are expressed as fractions of the maximum within-ROI radius, so the
    profile is comparable across cells of different sizes. Bin 0 is the core,
    bin 4 is the periphery.

    Returns a 5-element array. When added via extra_properties to
    regionprops_table, columns are roi_radial_profile-0 … roi_radial_profile-4.
    """
    n_bins = 5
    rows, cols = np.where(regionmask)
    if len(rows) == 0:
        return np.zeros(n_bins, dtype=float)

    cy, cx = rows.mean(), cols.mean()
    distances = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    max_dist = distances.max()

    if max_dist == 0:
        fill = float(intensity[rows[0], cols[0]])
        return np.full(n_bins, fill, dtype=float)

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    profile = np.zeros(n_bins, dtype=float)
    intensities = intensity[rows, cols]

    for i in range(n_bins):
        in_bin = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if in_bin.any():
            profile[i] = intensities[in_bin].mean()

    return profile
