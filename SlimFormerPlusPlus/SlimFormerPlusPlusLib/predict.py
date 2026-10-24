# SPDX-License-Identifier: Apache-2.0
"""SlimFormer++ inference with nnU-Net v2's predictor.

Preprocessing, sliding-window prediction and resampling are nnU-Net's own code
(``nnUNetPredictor``); only the network comes from this package, so no nnU-Net
trainer has to be registered and no CUDA extension is needed.

Command line (this is what the Slicer module runs in a separate process):

    PythonSlicer predict.py --model-dir DIR --input in.npz --output out.npz [--device auto]

``in.npz`` holds ``image`` (array[k, j, i], as from slicer.util.arrayFromVolume)
and ``ijk_to_ras`` (4x4). ``out.npz`` holds ``labels`` on the same voxel grid.
Lines starting with ``PROGRESS`` report the sliding-window progress.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from SlimFormerPlusPlusLib.reorient import from_model_axes, to_model_axes
else:
    from .reorient import from_model_axes, to_model_axes

CONFIGURATION = '3d_fullres'
CHECKPOINT = 'checkpoint_final.pth'

# Body bounding box (model axes k, j, i): air is about -1000 HU, fat about -100 HU.
BODY_HU = -500
BODY_GRID_MM = (4.0, 2.75, 2.75)      # coarse grid for the connected-component search
BODY_MARGIN_MM = (4.0, 11.0, 11.0)


def build_network(plans: dict, dataset_json: dict):
    if __package__ in (None, ''):
        from SlimFormerPlusPlusLib.model import SlimFormerPlusPlusAttention
    else:
        from .model import SlimFormerPlusPlusAttention
    arch = plans['configurations'][CONFIGURATION]['architecture']['arch_kwargs']
    if tuple(arch['strides'][0]) != (1, 1, 1):
        raise ValueError('Expected a full-resolution first stage')
    return SlimFormerPlusPlusAttention(len(dataset_json['channel_names']), len(dataset_json['labels']),
                                       first_stride=arch['strides'][1], level0_kernel=arch['kernel_sizes'][0])


def select_device(name: str):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no CUDA GPU is available to PyTorch')
    if name == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise RuntimeError('MPS was requested but is not available to PyTorch')
    return torch.device(name)


def logits_to_labels(predicted_logits, plans_manager, configuration_manager, label_manager, properties):
    """nnU-Net's ``convert_predicted_logits_to_segmentation_with_correct_shape`` with less memory.

    nnU-Net converts all class logits to float64 at once before resampling them to
    the original grid (16 classes of a typical CT: about 15 GB). Resampling works
    channel by channel, so here each class is resampled with the same function and
    kept only as a running maximum; ``>`` keeps the first maximum, like the
    ``argmax`` nnU-Net applies afterwards. The labels are identical.
    """
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    if label_manager.has_regions:
        raise NotImplementedError('region-based models are not supported')
    spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    shape = properties['shape_after_cropping_and_before_resampling']
    current_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == len(shape) \
        else [spacing[0], *configuration_manager.spacing]
    best = labels = None
    for c in range(predicted_logits.shape[0]):
        channel = configuration_manager.resampling_fn_probabilities(predicted_logits[c:c + 1], shape,
                                                                   current_spacing, spacing)[0]
        channel = channel.numpy() if hasattr(channel, 'numpy') else np.asarray(channel)
        if best is None:
            best, labels = channel.copy(), np.zeros(channel.shape, dtype=np.uint8)
        else:
            better = channel > best
            best[better] = channel[better]
            labels[better] = c
        del channel
    result = np.zeros(properties['shape_before_cropping'], dtype=np.uint8)
    result = insert_crop_into_image(result, labels, properties['bbox_used_for_cropping'])
    return result.transpose(plans_manager.transpose_backward)


def make_predictor(model_dir, device, tta=False, step_size=0.5, report=print):
    """nnUNetPredictor initialised with the SlimFormer++ network and weights in ``model_dir``."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    class ProgressPredictor(nnUNetPredictor):
        """Reports sliding-window progress as ``PROGRESS done total`` lines."""

        def _internal_get_sliding_window_slicers(self, image_size):
            slicers = super()._internal_get_sliding_window_slicers(image_size)
            self._progress = [0, len(slicers)]
            report(f'PROGRESS 0 {len(slicers)}')
            return slicers

        def predict_single_npy_array(self, input_image, image_properties, segmentation_previous_stage=None,
                                     output_file_truncated=None, save_or_return_probabilities=False):
            """nnU-Net's method, with the low-memory conversion to labels (``logits_to_labels``)."""
            if output_file_truncated is not None or save_or_return_probabilities:
                return super().predict_single_npy_array(input_image, image_properties, segmentation_previous_stage,
                                                        output_file_truncated, save_or_return_probabilities)
            from nnunetv2.inference import predict_from_raw_data as nnunet_predict
            ppa = nnunet_predict.PreprocessAdapterFromNpy(
                [input_image], [segmentation_previous_stage], [image_properties], [None], self.plans_manager,
                self.dataset_json, self.configuration_manager, num_threads_in_multithreaded=1,
                verbose=self.verbose)
            report('Preprocessing ...')
            dct = next(ppa)
            data = dct.pop('data')
            report('Predicting ...')
            predicted_logits = self.predict_logits_from_preprocessed_data(data).cpu()
            del data
            report('Resampling to the original image grid ...')
            return logits_to_labels(predicted_logits, self.plans_manager, self.configuration_manager,
                                    self.label_manager, dct['data_properties'])

        def _internal_maybe_mirror_and_predict(self, x):
            out = super()._internal_maybe_mirror_and_predict(x)
            p = getattr(self, '_progress', None)
            if p is not None:
                p[0] += 1
                report(f'PROGRESS {p[0]} {p[1]}')
            return out

    model_dir = Path(model_dir)
    plans = json.loads((model_dir / 'plans.json').read_text())
    dataset_json = json.loads((model_dir / 'dataset.json').read_text())
    checkpoint = torch.load(model_dir / 'fold_0' / CHECKPOINT, map_location='cpu', weights_only=True)
    network = build_network(plans, dataset_json)
    network.load_state_dict(checkpoint['network_weights'], strict=True)
    plans_manager = PlansManager(plans)
    with contextlib.redirect_stdout(io.StringIO()):  # nnU-Net prints a notice for every non-CUDA device
        predictor = ProgressPredictor(tile_step_size=step_size, use_gaussian=True, use_mirroring=tta,
                                      perform_everything_on_device=device.type == 'cuda', device=device,
                                      verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(network, plans_manager, plans_manager.get_configuration(CONFIGURATION),
                                    [checkpoint['network_weights']], dataset_json,
                                    checkpoint.get('trainer_name', 'SlimFormerPlusPlusAttention'),
                                    tuple(checkpoint.get('inference_allowed_mirroring_axes', (0, 1, 2))))
    return predictor, dataset_json


def body_box(image, spacing, min_extent_mm=(0.0, 0.0, 0.0)):
    """Bounding box (tuple of slices) of the body in a CT image, with a margin.

    nnU-Net crops only to the nonzero region, so the air around the patient and the
    table would be tiled by the sliding window too. The body is the largest connected
    region above ``BODY_HU`` on a coarse grid, after one erosion that detaches it from
    the table. The box is enlarged to at least ``min_extent_mm`` where the image allows,
    so that nnU-Net does not pad the cropped image where the full image has air. If no
    region is found (or the image is not in HU), the box is the whole image.
    """
    from scipy import ndimage
    spacing = np.asarray(spacing, dtype=float)
    shape = np.array(image.shape)
    step = np.maximum(1, np.round(np.array(BODY_GRID_MM) / spacing)).astype(int)
    mask = ndimage.binary_erosion(image[::step[0], ::step[1], ::step[2]] > BODY_HU)
    components, n = ndimage.label(mask)
    if n == 0:
        return tuple(slice(0, int(s)) for s in shape)
    largest = ndimage.find_objects(components)[int(np.bincount(components.ravel())[1:].argmax())]
    margin = np.ceil(np.array(BODY_MARGIN_MM) / spacing).astype(int)
    lo = np.array([s.start for s in largest]) * step - step - margin       # one coarse cell for the erosion
    hi = (np.array([s.stop for s in largest]) + 1) * step + margin
    lo, hi = np.maximum(lo, 0), np.minimum(hi, shape)
    grow = np.maximum(np.ceil(np.asarray(min_extent_mm) / spacing).astype(int) - (hi - lo), 0)
    lo, hi = lo - grow // 2, hi + grow - grow // 2
    shift = np.maximum(-lo, 0) - np.maximum(hi - shape, 0)                 # keep the enlarged box inside
    lo, hi = np.maximum(lo + shift, 0), np.minimum(hi + shift, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def patch_extent_mm(predictor):
    """Physical size of the network's patch along the image axes (k, j, i)."""
    cm = predictor.configuration_manager
    extent = [p * s for p, s in zip(cm.patch_size, cm.spacing)]          # in nnU-Net's transposed axis order
    out = [0.0] * len(extent)
    for t, axis in enumerate(predictor.plans_manager.transpose_forward):
        out[axis] = extent[t]
    return out


def segment(predictor, image_kji: np.ndarray, ijk_to_ras, crop_body=True, report=None) -> np.ndarray:
    """Label map on the input voxel grid (array[k, j, i], uint8).

    With ``crop_body`` only the body's bounding box is segmented; outside it the
    labels are background.
    """
    image, spacing, mapping = to_model_axes(image_kji, ijk_to_ras)
    box = tuple(slice(0, s) for s in image.shape)
    if crop_body:
        box = body_box(image, spacing, patch_extent_mm(predictor))
        if report:
            cropped = [s.stop - s.start for s in box]
            report(f'Body region: {cropped[::-1]} of {list(image.shape[::-1])} voxels '
                   f'({100 * np.prod(cropped) / np.prod(image.shape):.0f}% of the volume)')
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[box] = predictor.predict_single_npy_array(np.ascontiguousarray(image[box])[None].astype(np.float32, copy=False),
                                                     {'spacing': spacing}, None, None, False)
    return from_model_axes(labels, mapping)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu', 'mps'))
    p.add_argument('--tta', action='store_true', help='mirroring test-time augmentation (8x slower)')
    p.add_argument('--step-size', type=float, default=0.5)
    p.add_argument('--threads', type=int, default=0, help='CPU threads (0: PyTorch default)')
    p.add_argument('--no-crop-body', dest='crop_body', action='store_false',
                   help='segment the whole field of view instead of the body bounding box')
    a = p.parse_args(argv)

    import torch
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    device = select_device(a.device)
    print(f'Device: {device}; PyTorch {torch.__version__}; threads {torch.get_num_threads()}', flush=True)
    start = time.time()
    data = np.load(a.input)
    image, ijk_to_ras = data['image'], data['ijk_to_ras']
    report = lambda s: print(s, flush=True)  # noqa: E731
    predictor, _ = make_predictor(a.model_dir, device, a.tta, a.step_size, report=report)
    labels = segment(predictor, image, ijk_to_ras, crop_body=a.crop_body, report=report)
    np.savez_compressed(a.output, labels=labels)
    print(f'Done in {time.time() - start:.1f} s', flush=True)


if __name__ == '__main__':
    main()
