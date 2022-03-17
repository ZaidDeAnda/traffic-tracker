import matplotlib.pyplot as plt
import numpy as np


def plot_image(image, title=""):
    figure, axes = plt.subplots(1, 1, figsize=(5, 4), facecolor="white")
    axes.set_title(title)
    axes.imshow(image)


def cm_to_inch(value):
    return value / 2.54


def plot_image_huge(image, width=78, height=44, title="", grid=None, color=''):

    width = cm_to_inch(width)
    height = cm_to_inch(height)

    figure, axes = plt.subplots(1, 1, figsize=(width, height), facecolor="white")
    if grid:
        plt.grid(color="red")

    height = image.shape[0]
    width = image.shape[1]

    x_axis_labels = np.arange(0, width, 10)
    y_axis_labels = np.arange(0, height, 10)

    axes.set_xticks(x_axis_labels)
    axes.set_yticks(y_axis_labels)

    axes.tick_params(
        labelbottom=True,
        labeltop=True,
        labelleft=True,
        labelright=True,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )

    axes.set_title(title)
    if color:
        axes.imshow(image, cmap = color)
    else:
        axes.imshow(image)


def plot_images(
    nested_lists_labels: list, nested_list_images: list = None, show: int = None
):

    if not nested_list_images:
        print("nested_list_images is required")
        return

    rows = show or len(nested_list_images)
    cols = len(nested_list_images[0])

    figure, axes = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 4), facecolor="white"
    )

    for row, list_images in enumerate(nested_list_images):

        if row == show:
            break

        for col, image in enumerate(list_images):

            image_name = nested_lists_labels[row][col]

            if rows == 1 and cols == 1:
                axes.set_title(image_name)
                axes.imshow(image)

            elif rows > 1 and cols == 1:
                axes[row].set_title(image_name)
                axes[row].imshow(image)

            elif rows == 1 and cols > 1:
                axes[col].set_title(image_name)
                axes[col].imshow(image)

            elif rows > 1 and cols > 1:
                axes[row, col].set_title(image_name)
                axes[row, col].imshow(image)

            else:
                print("check plot file")
