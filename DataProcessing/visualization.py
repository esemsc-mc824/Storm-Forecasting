import os
import PIL
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import pearsonr
import matplotlib.animation as animation
from IPython.display import display, HTML
from PIL import Image
from .data_loading import load_event, load_multiple_events

def make_gif(outfile, files, fps=10, loop=0):
    """
    Creates and saves an animated GIF from a list of image files.

    Parameters
    ----------
    outfile : str
        Output file path where the GIF will be saved.
    files : list of str
        List of image file paths to be included in the GIF.
    fps : int, optional
        Frames per second for the GIF animation (default is 10).
    loop : int, optional
        Number of times the GIF should loop. Use 0 for infinite looping (default is 0).

    Returns
    -------
    IPython.display.Image
        The generated GIF, ready for display in Jupyter Notebook.

    Raises
    ------
    FileNotFoundError
        If any of the image files do not exist.
    """
    imgs = [PIL.Image.open(file) for file in files]
    imgs[0].save(
        fp=outfile,
        format='gif',
        append_images=imgs[1:],
        save_all=True,
        duration=int(1000/fps),
        loop=loop
    )
    im = IPython.display.Image(filename=outfile)
    im.reload()
    return im

def plot_event(event_id, output_gif=False, save_gif=False):
    """
    Helper function for plotting an event's satellite images and lightning data.

    Parameters
    ----------
    event_id : str
        The ID of the event to plot.
    output_gif : bool, optional
        If True, generates a GIF of the event frames (default is False).
    save_gif : bool, optional
        If True, saves the generated GIF to disk. If False, deletes the GIF after displaying (default is False).

    Returns
    -------
    IPython.display.Image or None
        The generated GIF if `output_gif` is True, otherwise None.

    Raises
    ------
    KeyError
        If the event_id is not found in the dataset.
    """
    event = load_event(event_id)
    t = event["lght"][:,0]  # Time of lightning strike (in seconds relative to first frame)
    
    def plot_frame(ti):
        """
        Plots a single frame of the event.

        Parameters
        ----------
        ti : int
            Frame index to plot.
        """
        f = (t >= ti*5*60 - 2.5*60) & (t < ti*5*60 + 2.5*60)  # Lightning strikes in current frame
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"Event: {event_id}, Frame: {ti}, Time: {ti*5} min")
        axs[0].imshow(event["vis"][:,:,ti], vmin=0, vmax=10000, cmap="grey")
        axs[0].set_title('Visible')
        axs[1].imshow(event["ir069"][:,:,ti], vmin=-8000, vmax=-1000, cmap="viridis")
        axs[1].set_title('Infrared (Water Vapor)')
        axs[2].imshow(event["ir107"][:,:,ti], vmin=-7000, vmax=2000, cmap="inferno")
        axs[2].set_title('Infrared (Cloud/Surface Temperature)')
        axs[3].imshow(event["vil"][:,:,ti], vmin=0, vmax=255, cmap="turbo")
        axs[3].set_title('Radar (Vertically Integrated Liquid)')
        axs[3].scatter(event["lght"][f,3], event["lght"][f,4], marker="x", s=30, c="tab:red")
        axs[3].set_xlim(0,384)
        axs[3].set_ylim(384,0)
        if output_gif:
            file = f"_temp_{event_id}_{ti}.png"
            fig.savefig(file, bbox_inches="tight", dpi=150, pad_inches=0.02, facecolor="white")
            plt.close()
        else:
            plt.show()
    
    if output_gif:
        for ti in range(event["vis"].shape[-1]):
            plot_frame(ti)
        gif_path = f"{event_id}.gif"
        make_gif(gif_path, [f"_temp_{event_id}_{ti}.png" for ti in range(event["vis"].shape[-1])])
        for ti in range(event["vis"].shape[-1]):
            os.remove(f"_temp_{event_id}_{ti}.png")
        im = IPython.display.Image(filename=gif_path)
        display(im)
        if not save_gif:
            os.remove(gif_path)
        return im
    else:
        # Plot selected frames for quick visualization
        selected_frames = [0, event["vis"].shape[-1]//2, event["vis"].shape[-1]-1]
        for ti in selected_frames:
            plot_frame(ti)
        return None

def check_correlation(vil_data):
    """
    Checks the correlation between consecutive VIL (Vertically Integrated Liquid) image frames.

    Parameters
    ----------
    vil_data : numpy.ndarray or torch.Tensor
        The VIL image data with shape (frames, height, width).

    Returns
    -------
    None
        Displays a heatmap of the correlation matrix.

    Raises
    ------
    ValueError
        If vil_data does not have three dimensions.
    """
    if vil_data.ndim != 3:
        raise ValueError("vil_data must have shape (frames, height, width).")
    
    vil_flat = vil_data.reshape(vil_data.shape[0], -1)  # Reshape to (frames, pixels)
    correlation_matrix = np.corrcoef(vil_flat)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between VIL Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    plt.show()

def plot_time_correlation(event):
    """
    Plots the time autocorrelation matrix for different image channels of an event.

    Parameters
    ----------
    event : dict
        Dictionary containing image data for the event with keys ['vis', 'ir069', 'ir107', 'vil'].

    Returns
    -------
    None
        Displays the correlation matrices in a grid of subplots.

    Raises
    ------
    KeyError
        If any of the required image types are missing in the event.
    """
    channels = ['vis', 'ir069', 'ir107', 'vil']
    num_frames = event['vis'].shape[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Time-lagged Correlation Matrices for Different Channels", fontsize=16)

    for idx, ch in enumerate(channels):
        if ch not in event:
            raise KeyError(f"Image type '{ch}' not found in the event data.")
        row, col = idx // 2, idx % 2
        corr_matrix = np.zeros((num_frames, num_frames))

        for i in range(num_frames):
            for j in range(num_frames):
                corr, _ = pearsonr(event[ch][:, i].flatten(), event[ch][:, j].flatten())
                corr_matrix[i, j] = corr

        sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, ax=axes[row, col])
        axes[row, col].set_title(f"{ch.upper()} Channel")
        axes[row, col].set_xlabel("Frame")
        axes[row, col].set_ylabel("Frame")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_lightning_distribution(events_data, event_ids=None, bins=50):
    """
    Plots the distribution of lightning occurrences over time for multiple events.

    Parameters
    ----------
    events_data : dict
        Dictionary where keys are event IDs and values are pandas DataFrames of lightning data.
    event_ids : list of str, optional
        List of event IDs to plot. If None, plots all available events (default is None).
    bins : int, optional
        Number of bins for the histogram (default is 50).

    Returns
    -------
    None
        Displays the histogram plot.

    Raises
    ------
    ValueError
        If any of the specified event_ids are not found in events_data.
    """
    plt.figure(figsize=(12, 6))

    if event_ids is None:
        event_ids = list(events_data.keys())

    for event_id in event_ids:
        if event_id in events_data:
            lightning_df = events_data[event_id]
            sns.histplot(lightning_df["t"], bins=bins, kde=True, label=f"Event {event_id}", alpha=0.6)
        else:
            print(f"Warning: Event ID {event_id} not found in events_data.")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of Lightning Flashes")
    plt.title("Lightning Occurrence Over Time Across Multiple Events")
    plt.legend()
    plt.show()

def plot_lightning_geographic_distribution(events_data, event_ids):
    """
    Plots the geographic distribution of lightning for multiple events.

    Parameters
    ----------
    events_data : dict
        Dictionary where keys are event IDs and values are pandas DataFrames of lightning data.
    event_ids : list of str
        List of event IDs to plot.

    Returns
    -------
    None
        Displays the geographic scatter plot.

    Raises
    ------
    ValueError
        If any of the specified event_ids are not found in events_data.
    """
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(event_ids))

    for i, event_id in enumerate(event_ids):
        if event_id in events_data:
            lightning_df = events_data[event_id]
            plt.scatter(
                lightning_df["lon (deg)"],
                lightning_df["lat (deg)"],
                alpha=0.5,
                color=colors[i],
                s=5,
                label=f"Event {event_id}"
            )
        else:
            print(f"Warning: Event ID {event_id} not found in data.")

    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.title("Geographic Distribution of Lightning Events")
    plt.legend()
    plt.show()

def create_visualizations(real_frames, pred_frames, figsize=(18, 6), interval=500):
    """
    Creates an animated visualization comparing real, predicted, and difference frames.
    Also plots the frame-by-frame L1 loss after displaying the animation.

    Parameters
    ----------
    real_frames : numpy.ndarray or torch.Tensor
        Ground truth frames with shape (T, H, W).
    pred_frames : numpy.ndarray or torch.Tensor
        Predicted frames with shape (T, H, W).
    figsize : tuple, optional
        Figure size for the animation (default is (18, 6)).
    interval : int, optional
        Animation frame interval in milliseconds (default is 500).

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The generated animation object.

    Raises
    ------
    ValueError
        If the number of real and predicted frames do not match.
    """
    T_real, H_real, W_real = real_frames.shape
    T_pred, H_pred, W_pred = pred_frames.shape

    if T_real != T_pred:
        raise ValueError("Mismatch in the number of real and predicted frames.")

    # Ensure resolution is consistent
    if (H_real, W_real) != (H_pred, W_pred):
        raise ValueError("Mismatch in frame dimensions between real and predicted frames.")

    # Compute absolute difference
    diff_frames = np.abs(real_frames - pred_frames)

    # Compute L1 loss per frame
    l1_loss_per_frame = np.mean(diff_frames, axis=(1, 2))  # Mean absolute error per frame

    # Create figure and subplots for animation
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Set titles
    axs[0].set_title("Ground Truth")
    axs[1].set_title("Prediction")
    axs[2].set_title("Difference")

    # Remove axes
    for ax in axs:
        ax.axis('off')

    # Initialize images
    im_real = axs[0].imshow(real_frames[0], cmap='jet', vmin=real_frames.min(), vmax=real_frames.max())
    im_pred = axs[1].imshow(pred_frames[0], cmap='jet', vmin=real_frames.min(), vmax=real_frames.max())
    im_diff = axs[2].imshow(diff_frames[0], cmap='gray', vmin=diff_frames.min(), vmax=diff_frames.max())

    # Update function for animation
    def update(frame_idx):
        im_real.set_data(real_frames[frame_idx])
        im_pred.set_data(pred_frames[frame_idx])
        im_diff.set_data(diff_frames[frame_idx])
        return [im_real, im_pred, im_diff]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=T_real, interval=interval, blit=True, repeat=True
    )

    # Close the static figure to prevent it from displaying
    plt.close(fig)

    # Convert animation to HTML
    html_anim = ani.to_jshtml()

    # Display animation
    display(HTML(html_anim))

    # Plot L1 loss after displaying animation
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, T_real + 1), l1_loss_per_frame, marker='o', linestyle='-', color='red')
    plt.title("Frame-by-Frame L1 Loss")
    plt.xlabel("Frame Number")
    plt.ylabel("L1 Loss")
    plt.xticks(np.arange(1, T_real + 1))
    plt.grid(True)
    plt.show()

    return ani
