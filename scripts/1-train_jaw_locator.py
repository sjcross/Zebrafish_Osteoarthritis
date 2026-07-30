"""
Train a model to locate the zebrafish jaw.

The jaw segmentation model doesn't take the whole scan as an input - instead, it
takes a cropped out sub-image just containing the fish head.

Therefore, we first need to locate the fish head in our scans - we do this, again
with a machine learning model.

"""

import pathlib
import argparse
import warnings

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

from fishlib.images import io
from fishlib.images.transform import crop
from fishlib.util import util, files
from fishlib.localisation import data, plotting, model
from fishlib.visualisation.training import plot_losses, plot_loss_axis


def _create_downsampled_dicoms(
    target_size: tuple[int, int, int],
    dicom_paths: list[pathlib.Path],
    downsampled_paths: list[pathlib.Path],
) -> None:
    """
    Read in the image/labels from the full-resolution DICOM files,
    downsample them then create new downsampled DICOM files
    """
    pbar = tqdm(zip(dicom_paths, downsampled_paths), total=len(dicom_paths))

    for in_path, out_path in zip(dicom_paths, downsampled_paths):
        if not out_path.exists():
            out_path.parent.mkdir(exist_ok=True, parents=True)

            pbar.set_description(f"Reading {in_path.name}")
            img, label = io.read_dicom(in_path)

            pbar.set_description(f"Downsampling {in_path.name}")
            img, label = data.downsample(img, label, target_size)

            # Create a dicom and save it
            dicom = data.write_dicom(img, label, out_path)
        pbar.update(1)


def _train_test_split(
    downsampled_paths: list[pathlib.Path], dicom_paths: list[pathlib.Path]
) -> tuple[list[pathlib.Path], list[pathlib.Path], pathlib.Path, pathlib.Path]:
    """
    Get the:
     - training data (list of paths)
     - validation data (list of paths)
     - full-res test data (one path)
     - downsampled test data (one path)
    """
    n_paths = len(downsampled_paths)
    assert n_paths == len(
        dicom_paths
    ), f"{len(downsampled_paths)=} but {len(dicom_paths)=}"

    # Leave the last one for testing
    train_paths = downsampled_paths[: n_paths - 2]
    val_paths = downsampled_paths[n_paths - 2 : -1]

    test_path = dicom_paths[-1]
    downsampled_test_path = downsampled_paths[-1]

    return train_paths, val_paths, test_path, downsampled_test_path


def _savefig(fig: plt.Figure, path: pathlib.Path, *, verbose: bool) -> None:
    """
    Helper function for saving figures

    Also closes the figure
    """
    if verbose:
        print(f"Saving figure to {path}")
    fig.savefig(path)
    plt.close(fig)


def _dicom_paths(config: dict) -> list[pathlib.Path]:
    """
    The paths to the training DICOMs
    """
    input_dirs = [pathlib.Path(d) for d in config["dicom_dirs"]]
    return sorted(
        [path for input_dir in input_dirs for path in input_dir.glob("**/*.dcm") if "downsampled_dicoms" not in path.parts]
    )


def main(model_name: str, debug_plots: bool, dont_shrink_heatmap: bool) -> None:
    """
    Read (cached) downsampled dicoms (caching them first if required),
    init a model and train it to localise the jaw.

    The jaw centre is the centroid of the segmentation mask; we will use a heatmap
    with a gradually shrinking kernel to train the model. Then we will recover
    the jaw centre from the heatmap by convolving to find its centre.

    """
    # accidentally named my variables confusingly
    shrink_heatmap = not dont_shrink_heatmap

    # The configuration file contains a weird ugly dictionary that contains
    # all the configuration for the jaw location
    config = util.userconf()["jaw_loc_config"]

    # Find where the inputs are, and if necessary create the downsampled dicoms
    dicom_paths = _dicom_paths(config)
    downsampled_paths = [data.downsampled_dicom_path(p) for p in dicom_paths]

    if not all(p.exists() for p in downsampled_paths):
        _create_downsampled_dicoms(
            target_size=config["downsampled_dicom_size"],
            dicom_paths=dicom_paths,
            downsampled_paths=downsampled_paths,
        )

    # This checks that we haven't accidentally messed something up with the paths
    parent_dirs = set(p.parent for p in downsampled_paths)
    assert len(parent_dirs) == len(
        config["dicom_dirs"]
    ), "Should have the same number of downsampled dicom dirs as input dicom dirs"
    # TODO delete; can't tell what this is doing
    for parent_dir in parent_dirs:
        parent_dir.mkdir(parents=True, exist_ok=True)

    # Define where the outputs should go - the model path...
    model_path = files.jaw_locator_model_path(model_name)
    if model_path.exists():
        raise FileExistsError(
            f"Model already exists at {model_path}, please delete it or use a different name."
        )
    # ...and the directory for plots
    out_dir = model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the training and validation data
    # And testing data - both for the full-sized image for an end-to-end test,
    # and the downsampled image
    train_paths, val_paths, full_res_test_path, downsampled_test_path = (
        _train_test_split(downsampled_paths, dicom_paths)
    )
    print(len(train_paths), "train, ", len(val_paths), "val")
    
    assert len(train_paths) < config["batch_size"], "Batch size shouldn't be larger than training data size"

    if len(train_paths) < config["batch_size"]:
        raise ValueError("Batch size shouldn't be larger than training data size")
        
    # Set up training data heatmaps
    train_imgs, train_labels = zip(*[io.read_dicom(p) for p in train_paths])
    train_data = data.HeatmapDataset(
        images=train_imgs,
        masks=train_labels,
        sigma=config["initial_kernel_size"],
        augment=True,
    )

    val_imgs, val_labels = zip(*[io.read_dicom(p) for p in val_paths])
    val_data = data.HeatmapDataset(
        images=val_imgs,
        masks=val_labels,
        sigma=config["initial_kernel_size"],
        augment=False,
    )

    # Plot training and validation heatmaps
    if debug_plots:
        for dataset, name in zip([train_data, val_data], ["train", "val"]):
            img, label = dataset[0]
            fig, _ = plotting.plot_heatmap(img.unsqueeze(0), label.unsqueeze(0))
            _savefig(fig, out_dir / f"{name}_heatmap_example.png", verbose=True)

    net = model.get_model(config["device"])
    print(f"{sum(p.numel() for p in net.parameters() if p.requires_grad):,} params")
    train_metrics = model.train(
        net,
        train_data,
        val_data,
        config["learning_rate"],
        config["batch_size"],
        config["num_epochs"],
        config["n_workers"],
        config["device"],
        shrink_heatmap,
        out_dir,
    )
    with open(model_path, "wb") as f:
        torch.save(net.state_dict(), f)

    net = train_metrics.model
    train_losses = train_metrics.train_losses
    val_losses = train_metrics.val_losses

    # Plot losses
    fig = plot_losses(train_losses, val_losses)
    _savefig(fig, out_dir / "losses.png", verbose=debug_plots)

    if debug_plots:
        # Plot heatmaps for training + val data
        for dataset, name in zip([train_data, val_data], ["train", "val"]):
            img, _ = dataset[0]
            prediction = model.heatmap(net, img.squeeze().numpy())

            # Convert to tensor for plotting
            prediction = torch.tensor(prediction).unsqueeze(0).unsqueeze(0)
            fig, _ = plotting.plot_heatmap(img.unsqueeze(0), prediction)
            _savefig(fig, out_dir / f"{name}_heatmap_pred.png", verbose=True)

        # Plot the other metrics
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        plot_loss_axis(axes[0], train_metrics.train_kl, train_metrics.val_kl)
        plot_loss_axis(axes[1], train_metrics.train_dice, train_metrics.val_dice)
        plot_loss_axis(axes[2], train_metrics.train_mse, train_metrics.val_mse)
        for axis, title in zip(axes, ["KL", "Dice", "MSE"]):
            axis.set_title(title)
        _savefig(fig, out_dir / "metrics.png", verbose=True)

    # Read in the original and downsampled test data
    # We may want to plot the heatmap on the downsampled data (for debug)
    # Also plot the actual/predicted centre on the original size image
    test_img, test_label = io.read_dicom(full_res_test_path)
    downsampled_test_img, _ = io.read_dicom(downsampled_test_path)

    # Plot heatmap
    if debug_plots:
        predicted_heatmap = model.heatmap(net, downsampled_test_img)
        fig, _ = plotting.plot_heatmap(
            torch.tensor(downsampled_test_img.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0),
            torch.tensor(predicted_heatmap).unsqueeze(0).unsqueeze(0),
        )
        _savefig(fig, out_dir / "test_heatmap.png", verbose=True)

    # Find the predicted centroid
    predicted_centroid = model.predict_centroid(net, downsampled_test_img)
    if debug_plots:
        # Plot the centroid on the downsampled image
        fig, _ = plotting.plot_centroid(
            torch.tensor(downsampled_test_img.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
            predicted_centroid,
        )
        _savefig(fig, out_dir / "test_centroid_downsampled.png", verbose=True)

        # Plot the truth centroid
        truth_centroid = [int(x) for x in center_of_mass(test_label)]
        fig, _ = plotting.plot_centroid(
            torch.tensor(test_img.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
            truth_centroid,
        )
        _savefig(fig, out_dir / "test_centroid_truth.png", verbose=True)

    # Find the scale factor
    scaled_predicted_centroid = data.scale_prediction_up(
        predicted_centroid,
        data.scale_factor(test_img.shape, downsampled_test_img.shape),
    )

    # Plot the predicted centroid on the original image
    fig, _ = plotting.plot_centroid(
        torch.tensor(test_img.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        scaled_predicted_centroid,
    )
    _savefig(fig, out_dir / "test_centroid.png", verbose=debug_plots)

    # Crop using the prediction, save the image
    cropped = model.crop(
        net, test_img, config["downsampled_dicom_size"], config["crop_size"]
    )

    # Since we have the mask, we can also crop it and
    # plot it on the same image
    cropped_mask = crop(
        test_label,
        scaled_predicted_centroid,
        config["crop_size"],
        centred=True,
    )
    if cropped_mask.sum() > 1e-6:
        fig, _ = plotting.plot_heatmap(
            torch.tensor(cropped.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
            torch.tensor(cropped_mask).unsqueeze(0).unsqueeze(0),
        )
        _savefig(fig, out_dir / "test_cropped.png", verbose=debug_plots)
    else:
        warnings.warn("Cropped mask is empty, not plotting test_cropped.png")

    # We can also plot the cropped mask, with the jaw overlaid
    if debug_plots:
        true_cropped_img = crop(
            test_img, truth_centroid, config["crop_size"], centred=True
        )
        true_cropped_mask = crop(
            test_label, truth_centroid, config["crop_size"], centred=True
        )
        fig, _ = plotting.plot_heatmap(
            torch.tensor(true_cropped_img.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
            torch.tensor(true_cropped_mask).unsqueeze(0).unsqueeze(0),
        )
        _savefig(fig, out_dir / "truth_cropped.png", verbose=debug_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to locate the zebrafish jaw."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="example_locator",
    )
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        help="Plot the training data and downsampled testing data/heatmaps for test data."
        "Losses and upsampled point estimate on test data are always plotted",
    )

    parser.add_argument(
        "--dont-shrink-heatmap",
        action="store_true",
        help="Don't shrink the heatmap during training if the loss is low.",
    )

    main(**vars(parser.parse_args()))
