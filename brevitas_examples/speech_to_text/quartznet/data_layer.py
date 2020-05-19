# Adapted from https://github.com/NVIDIA/NeMo/blob/r0.9/collections/nemo_asr/
# Copyright (C) 2020 Xilinx (Giuseppe Franco)
# Copyright (C) 2019 NVIDIA CORPORATION.
#
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This package contains Neural Modules responsible for ASR-related
data layers.
"""
__all__ = ['AudioToTextDataLayer']

import copy
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
from functools import partial
from torch.utils.data import Dataset

from .parts.features import WaveformFeaturizer
from .parts.perturb import AudioAugmentor, perturbation_types
from .parts.dataset import (
    AudioDataset,
    AudioLabelDataset,
    fixed_seq_collate_fn,
    seq_collate_fn,
)
def pad_to(x, k=8):
    """Pad int value up to divisor of k.

    Examples:
        >>> pad_to(31, 8)
        32

    """

    return x + (x % k > 0) * (k - x % k)
def _process_augmentations(augmenter) -> AudioAugmentor:
    """Process list of online data augmentations.

    Accepts either an AudioAugmentor object with pre-defined augmentations,
    or a dictionary that points to augmentations that have been defined.

    If a dictionary is passed, must follow the below structure:
    Dict[str, Dict[str, Any]]: Which refers to a dictionary of string
    names for augmentations, defined in `asr/parts/perturb.py`.

    The inner dictionary may contain key-value arguments of the specific
    augmentation, along with an essential key `prob`. `prob` declares the
    probability of the augmentation being applied, and must be a float
    value in the range [0, 1].

    # Example in YAML config file

    Augmentations are generally applied only during training, so we can add
    these augmentations to our yaml config file, and modify the behaviour
    for training and evaluation.

    ```yaml
    AudioToSpeechLabelDataLayer:
        ...  # Parameters shared between train and evaluation time
        train:
            augmentor:
                shift:
                    prob: 0.5
                    min_shift_ms: -5.0
                    max_shift_ms: 5.0
                white_noise:
                    prob: 1.0
                    min_level: -90
                    max_level: -46
                ...
        eval:
            ...
    ```

    Then in the training script,

    ```python
    import copy
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with open(model_config) as f:
        params = yaml.load(f)

    # Train Config for Data Loader
    train_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    train_dl_params.update(params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]

    data_layer_train = nemo_asr.AudioToTextDataLayer(
        ...,
        **train_dl_params,
    )

    # Evaluation Config for Data Loader
    eval_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    eval_dl_params.update(params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]

    data_layer_eval = nemo_asr.AudioToTextDataLayer(
        ...,
        **eval_dl_params,
    )
    ```

    # Registering your own Augmentations

    To register custom augmentations to obtain the above convenience of
    the declaring the augmentations in YAML, you can put additional keys in
    `perturbation_types` dictionary as follows.

    ```python
    from nemo.collections.asr.parts import perturb

    # Define your own perturbation here
    class CustomPerturbation(perturb.Perturbation):
        ...

    perturb.register_perturbation(name_of_perturbation, CustomPerturbation)
    ```

    Args:
        augmenter: AudioAugmentor object or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.

    Returns: AudioAugmentor object
    """
    if isinstance(augmenter, AudioAugmentor):
        return augmenter

    if not type(augmenter) == dict:
        raise ValueError("Cannot parse augmenter. Must be a dict or an AudioAugmentor object ")

    augmenter = copy.deepcopy(augmenter)

    augmentations = []
    for augment_name, augment_kwargs in augmenter.items():
        prob = augment_kwargs.get('prob', None)

        if prob is None:
            raise KeyError(
                f'Augmentation "{augment_name}" will not be applied as '
                f'keyword argument "prob" was not defined for this augmentation.'
            )

        else:
            _ = augment_kwargs.pop('prob')

            if prob < 0.0 or prob > 1.0:
                raise ValueError("`prob` must be a float value between 0 and 1.")

            try:
                augmentation = perturbation_types[augment_name](**augment_kwargs)
                augmentations.append([prob, augmentation])
            except KeyError:
                raise KeyError(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

    augmenter = AudioAugmentor(perturbations=augmentations)
    return augmenter


# Ported from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_commands.py
class AudioToSpeechLabelDataLayer:
    """Data Layer for general speech classification.

    Module which reads speech recognition with target label. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their target labels. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "label": \
target_label_0, "offset": offset_in_sec_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "label": \
target_label_n, "offset": offset_in_sec_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speech recognition model.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        augmenter (AudioAugmentor or dict): Optional AudioAugmentor or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
        time_length (int): max seconds to consider in a batch # Pass this only for speaker recognition task
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str],
        batch_size: int,
        sample_rate: int = 16000,
        int_values: bool = False,
        num_workers: int = 0,
        shuffle: bool = True,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim_silence: bool = False,
        drop_last: bool = False,
        load_audio: bool = True,
        augmentor: Optional[Union[AudioAugmentor, Dict[str, Dict[str, Any]]]] = None,
        time_length: int = 0,
    ):
        super(AudioToSpeechLabelDataLayer, self).__init__()

        self._manifest_filepath = manifest_filepath
        self._labels = labels
        self._sample_rate = sample_rate

        if augmentor is not None:
            augmentor = _process_augmentations(augmentor)

        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

        dataset_params = {
            'manifest_filepath': manifest_filepath,
            'labels': labels,
            'featurizer': self._featurizer,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'trim': trim_silence,
            'load_audio': load_audio,
        }
        self._dataset = AudioLabelDataset(**dataset_params)

        self.num_classes = self._dataset.num_commands
        self.labels = self._dataset.labels
        # Set up data loader
        # if self._placement == DeviceType.AllGpu:
        #     logging.info("Parallelizing Datalayer.")
        #     sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        # else:
        sampler = None

        if time_length:
            collate_func = partial(fixed_seq_collate_fn, fixed_length=time_length * self._sample_rate)
        else:
            collate_func = partial(seq_collate_fn, token_pad_value=0)

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=collate_func,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


class AudioToTextDataLayer(nn.Module):
    """Data Layer for general ASR tasks.

    Module which reads ASR labeled data. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their transcripts. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "text": \
transcript_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "text": \
transcript_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (str): Dataset parameter.
            End of string symbol used for seq2seq models.
            Defaults to None.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.
    """

    def __init__(
            self, *,
            manifest_filepath,
            labels,
            batch_size,
            sample_rate=16000,
            int_values=False,
            bos_id=None,
            eos_id=None,
            pad_id=None,
            min_duration=0.1,
            max_duration=None,
            normalize_transcripts=True,
            trim_silence=False,
            load_audio=True,
            drop_last=False,
            shuffle=True,
            num_workers=4,
            placement='cpu',
            # perturb_config=None,
            **kwargs
    ):
        super().__init__()

        self._featurizer = WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=None)

        # Set up dataset
        dataset_params = {'manifest_filepath': manifest_filepath,
                          'labels': labels,
                          'featurizer': self._featurizer,
                          'max_duration': max_duration,
                          'min_duration': min_duration,
                          'normalize': normalize_transcripts,
                          'trim': trim_silence,
                          'bos_id': bos_id,
                          'eos_id': eos_id,
                          'logger': None,
                          'load_audio': load_audio}

        self._dataset = AudioDataset(**dataset_params)

        # Set up data loader
        if placement == 'cuda':
            print('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._dataset)
        else:
            sampler = None

        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(seq_collate_fn, token_pad_value=pad_id),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader

