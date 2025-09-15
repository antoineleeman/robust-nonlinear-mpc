import numpy as np


def rectangle_coordinates(origin, semi_width, semi_height):
    """
    Calculate the coordinates of a rectangle's corners.

    Parameters:
    origin (tuple): (x, y) coordinate of the rectangle's center.
    semi_width (float): Half the width of the rectangle.
    semi_height (float): Half the height of the rectangle.

    Returns:
    list: List of tuples representing the coordinates of the corners in the order
          [bottom-left, bottom-right, top-right, top-left].
    """
    x, y = origin

    # Calculate the coordinates of the four corners
    corners_rect = np.array([
        [x - semi_width, y - semi_height],  # Bottom-left
        [x + semi_width, y - semi_height],  # Bottom-right
        [x + semi_width, y + semi_height],  # Top-right
        [x - semi_width, y + semi_height],  # Top-left
    ])

    return corners_rect


# Example usage:
origin = (3, 4)
semi_width = 2
semi_height = 1.5

corners = rectangle_coordinates(origin, semi_width, semi_height)
print("Corners of the rectangle:", corners)
