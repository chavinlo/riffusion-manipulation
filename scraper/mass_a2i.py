import os
import shutil
import pathlib

import io
from imohash import hashfile
import numpy as np
from PIL import Image
import pydub
from scipy.io import wavfile
import torch
import torchaudio
import argparse

#we need to manipulate the funcs

def spectrogram_image_from_wav(wav_bytes: io.BytesIO, max_volume: float = 50, power_for_image: float = 0.25, ms_duration: int = 5119) -> Image.Image:
    """
    Generate a spectrogram image from a WAV file.
    """
    # Read WAV file from bytes
    sample_rate, waveform = wavfile.read(wav_bytes)

    #sample_rate = 44100  # [Hz]
    clip_duration_ms = ms_duration  # [ms]

    bins_per_image = 512
    n_mels = int(512)
    mel_scale = True

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    num_samples = int(512 / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    # Compute spectrogram from waveform
    Sxx = spectrogram_from_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        mel_scale=mel_scale,
        n_mels=n_mels,
    )

    # Convert spectrogram to image
    image = image_from_spectrogram(Sxx, max_volume=max_volume, power_for_image=power_for_image)

    return image

def spectrogram_from_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mel_scale: bool = True,
    n_mels: int = 512,
) -> np.ndarray:
    """
    Compute a spectrogram from a waveform.
    """

    spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=None,
        hop_length=hop_length,
        win_length=win_length,
    )

    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
    Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]

    Sxx_mag = np.abs(Sxx_complex)

    if mel_scale:
        mel_scaler = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        )

        Sxx_mag = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()

    return Sxx_mag

def image_from_spectrogram(
        data: np.ndarray,
        max_volume: float = 50,
        power_for_image: float = 0.25
) -> Image.Image:
    data = np.power(data, power_for_image)
    data = data / (max_volume / 255)
    data = 255 - data
    data = data[::-1]
    image = Image.fromarray(data.astype(np.uint8))
    return image

PATH = "videos"
OUTPATH = "spectogram_dataset"
os.makedirs(OUTPATH, exist_ok=True)
list_dir = os.listdir(PATH)

print("number of files in dir:", len(list_dir))

for file in list_dir:
    filename, ext = os.path.splitext(file)
    print(filename)
    if ext == ".wav":
        txt_pair_exist = os.path.isfile(PATH + "/" + filename + ".txt")
        if txt_pair_exist:
            pass
        else:
            os.remove(PATH + "/" + file)

list_dir = os.listdir(PATH)

max_volume = 100
power_for_image = 0.25

no_processed_files = 0

for file in list_dir:
    no_processed_files += 1
    print("PROCESSED ", no_processed_files, "/", len(list_dir))
    filename, ext = os.path.splitext(file)
    if ext == ".wav":
        #File is Audio
        audio_hash = hashfile(PATH + "/" + file, hexdigest=True)
        audio = pydub.AudioSegment.from_file(PATH + "/" + file)

        # Convert to mono and set frame rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(44100)

        length_in_ms = len(audio)
        print("ORIGINAL AUDIO LENGTH IN MS:", length_in_ms)
        
        #Calculate how many blocks can be generated
        chunk_size = 5119
        num_chunks, remaining_ms = divmod(length_in_ms, chunk_size)

        audio_chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = audio[start:end]
            audio_chunks.append(chunk)

        no_chunks = len(audio_chunks)
        print("GENERATED", no_chunks, "AUDIO CHUNKS, AND DROPPED", remaining_ms, "MS")

        wav_bytes_chunks = []

        for i, chunk in enumerate(audio_chunks):
            wav_bytes = io.BytesIO()
            chunk.export(wav_bytes, format="wav")
            wav_bytes.seek(0)
            wav_bytes_chunks.append(wav_bytes)
            spec_image = spectrogram_image_from_wav(wav_bytes, max_volume=max_volume, power_for_image=power_for_image, ms_duration=chunk_size)
            spec_image.save(OUTPATH + "/" + audio_hash + "_chunk_" + str(i) + ".png")
            shutil.copy(PATH + "/" + filename + ".txt", OUTPATH + "/" + audio_hash + "_chunk_" + str(i) + ".txt")

