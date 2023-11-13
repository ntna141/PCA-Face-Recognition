'''data_transformations.py
YOUR NAME HERE
Performs translation, scaling, and rotation transformations on data
CS 251: Data Analysis and Visualization
Fall 2023

NOTE: All functions should be implemented from scratch without loops using basic NumPy — no high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    column_means = np.mean(data, axis=0)

    centered_data = data - column_means

    return centered_data


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''

    # Find out what the column index is for the requested header
    radians = np.deg2rad(degrees)
    
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(radians), -np.sin(radians)],
            [0, np.sin(radians), np.cos(radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(radians), 0, np.sin(radians)],
            [0, 1, 0],
            [-np.sin(radians), 0, np.cos(radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(radians), -np.sin(radians), 0],
            [np.sin(radians), np.cos(radians), 0],
            [0, 0, 1]
        ])

    return rotation_matrix
