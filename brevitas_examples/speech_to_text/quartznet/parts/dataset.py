# Audio dataset and corresponding functions taken from Patter  https://github.com/ryanleary/patter
# Adapted from https://github.com/NVIDIA/NeMo/tree/r0.9/collections/nemo_asr
# MIT License
# Copyright (c) 2020 Xilinx (Giuseppe Franco)
# Copyright (c) 2019 NVIDIA Corporation
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.utils.data import Dataset

from .manifest import ManifestBase, ManifestEN
from .collections import ASRSpeechLabel, ASRAudioText
from .parsers import make_parser

def seq_collate_fn(batch, token_pad_value=0):
    """collate batch of audio sig, audio len, tokens, tokens len

    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).

    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(
                tokens_i, pad, value=token_pad_value)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def fixed_seq_collate_fn(batch, fixed_length=16000):
    """collate batch of audio sig, audio len, tokens, tokens len

    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
        fixed_length (Optional[int]): length of input signal to be considered

    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)

    has_audio = audio_lengths[0] is not None
    fixed_length = min(fixed_length, max(audio_lengths))

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, _ in batch:
        if has_audio:
            sig_len = sig_len.item()
            chunck_len = sig_len - fixed_length
            if chunck_len < 0:
                # pad = (0,fixed_length-sig_len)
                # signal = torch.nn.functional.pad(sig,pad)
                repeat = fixed_length // sig_len
                rem = fixed_length % sig_len
                sub = sig[-rem:] if rem > 0 else torch.tensor([])
                rep_sig = torch.cat(repeat * [sig])
                signal = torch.cat((rep_sig, sub))
                # print(sig_len,repeat,rem,len(sub),len(rep_sig),len(signal))
            else:
                start_idx = torch.randint(0, chunck_len, (1,)) if chunck_len else torch.tensor(0)
                end_idx = start_idx + fixed_length
                signal = sig[start_idx:end_idx]

            audio_signal.append(signal)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def audio_seq_collate_fn(batch):
    """
    Collate a batch (iterable of (sample tensor, label tensor) tuples) into
    properly shaped data tensors
    :param batch:
    :return: inputs (batch_size, num_features, seq_length), targets,
    input_lengths, target_sizes
    """
    # sort batch by descending sequence length (for packed sequences later)
    batch.sort(key=lambda x: x[0].size(0), reverse=True)

    # init tensors we need to return
    inputs = []
    input_lengths = []
    target_sizes = []
    targets = []
    metadata = []

    # iterate over minibatch to fill in tensors appropriately
    for i, sample in enumerate(batch):
        input_lengths.append(sample[0].size(0))
        inputs.append(sample[0])
        target_sizes.append(len(sample[1]))
        targets.extend(sample[1])
        metadata.append(sample[2])
    targets = torch.tensor(targets, dtype=torch.long)
    inputs = torch.stack(inputs)
    input_lengths = torch.stack(input_lengths)
    target_sizes = torch.stack(target_sizes)

    return inputs, targets, input_lengths, target_sizes, metadata


class AudioDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:

    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
    """
    def __init__(
            self,
            manifest_filepath,
            labels,
            featurizer,
            max_duration=None,
            min_duration=None,
            max_utts=0,
            blank_index=-1,
            unk_index=-1,
            normalize=True,
            trim=False,
            bos_id=None,
            eos_id=None,
            logger=False,
            load_audio=True,
            parser='en',
    ):
        self.collection = ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=make_parser(
                labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize,
            ),
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.load_audio = load_audio
        # if logger:
        #     logger.info(
        #         "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} "
        #         "hours.".format(
        #             self.manifest.duration / 3600,
        #             self.manifest.filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim,
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t, tl = sample.text_tokens, len(sample.text_tokens)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.collection)

class AudioLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:

    {"audio_filepath": "/path/to/audio.wav", "label":
    "label", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "label": "label",
    "offset": 301.75, "duration": 0.82}

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        trim: Boolean flag whether to trim the audio
        load_audio: Boolean flag indicate whether do or not load audio
    """

    def __init__(
        self,
        manifest_filepath,
        featurizer,
        labels=None,
        max_duration=None,
        min_duration=None,
        trim=False,
        load_audio=True,
    ):
        self.collection = ASRSpeechLabel(
            manifests_files=manifest_filepath.split(','), min_duration=min_duration, max_duration=max_duration,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.load_audio = load_audio

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_commands = len(self.labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            print(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t = self.label2id[sample.label]
        tl = 1  # For compatibility with collate_fn used later

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.collection)
