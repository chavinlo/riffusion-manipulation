import io

import numpy as np
from PIL import Image
import pydub
from scipy.io import wavfile
import torch
import torchaudio
import argparse
from typing import List
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input file to process, anything that FFMPEG supports, but wav and mp3 are recommended")
parser.add_argument("-o", "--output", type=str, default="output", help="Output Folder")
parser.add_argument("-m", "--maxvol", type=int, default=100, help="Max Volume, 255 for identical results")
parser.add_argument("-p", "--powerforimage", type=float, default=0.25, help="Power for Image")
parser.add_argument("-n", "--nmels", type=int, default=512, help="n_mels to use for Image, basically HEIGHT. Higher = more fidelity")
parser.add_argument("-d", "--duration", type=int, default=5119, help="Duration of each chunk")
args = parser.parse_args()
# int and float checks
args.input = str(args.input)
args.output = str(args.output)
args.maxvol = int(args.maxvol)
args.powerforimage = float(args.powerforimage)
args.nmels = int(args.nmels)
args.duration = int(args.duration)

def spectrogram_image_from_wav(
    wav_bytes: io.BytesIO, 
    max_volume: float = 50, 
    power_for_image: float = 0.25, 
    ms_duration: int = 5119,
    nmels: int = 512) -> Image.Image:
    """
    Generate a spectrogram image from a WAV file.
    """
    # Read WAV file from bytes
    sample_rate, waveform = wavfile.read(wav_bytes)

    #sample_rate = 44100  # [Hz]
    clip_duration_ms = ms_duration  # [ms]

    bins_per_image = 512
    n_mels = nmels
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
    image = image_from_spectrogram(
        Sxx, 
        max_volume=max_volume, 
        power_for_image=power_for_image)

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
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    image = Image.fromarray(data.astype(np.uint8))
    return image

def spectrogram_images_from_file(
    filename: str, 
    max_volume: float = 50, 
    power_for_image: float = 0.25, 
    nmels: int = 512, 
    duration: int = 5119
        ) -> List[Image.Image]:
    """
    Generate a list of spectrogram images from an MP3 file.
    """
    # Load MP3 file into AudioSegment object
    audio = pydub.AudioSegment.from_file(filename)

    # Convert to mono and set frame rate
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(44100)

    # Calculate the number of 5 second intervals in the audio
    interval_count = len(audio) // duration

    print("CHUNKS TO PROCESS:", interval_count)

    # Initialize list to store spectrogram images
    spectrogram_images = []

    # Iterate over intervals and generate spectrogram images
    for i in range(interval_count):
        print("PROCESSED:", i, "/", interval_count)
        # Extract 5 second interval of audio data
        interval_audio = audio[i*duration:(i+1)*duration]

        # Convert to WAV and save as BytesIO object
        wav_bytes = io.BytesIO()
        interval_audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)

        # Generate spectrogram image from WAV file
        spectrogram_image = spectrogram_image_from_wav(
            wav_bytes, 
            max_volume=max_volume, 
            power_for_image=power_for_image,
            ms_duration=duration,
            nmels=nmels)

        # Add image to list
        spectrogram_images.append(spectrogram_image)

    # Check if there are any leftover seconds that are not a multiple of 5
    leftover_seconds = len(audio) % duration
    if leftover_seconds > 0:
        print("PROCESSING LEFTOVER CHUNK")
        # Extract the leftover interval of audio data
        interval_audio = audio[-leftover_seconds:]

        # Calculate amount of silent audio to add and combine
        add_ms = duration - leftover_seconds
        print("ON THE LAST CHUNK,", add_ms,"ms WILL BE SILENT")
        silence = pydub.AudioSegment.silent(
            duration=add_ms,
            frame_rate=44100
            )

        combined_segment = interval_audio + silence

        # Convert to WAV and save as BytesIO object
        wav_bytes = io.BytesIO()
        combined_segment.export(wav_bytes, format="wav")
        wav_bytes.seek(0)

        # Generate spectrogram image from WAV file
        spectrogram_image = spectrogram_image_from_wav(
            wav_bytes, 
            max_volume=max_volume, 
            power_for_image=power_for_image,
            ms_duration=duration,
            nmels=nmels)

        # Add image to list
        spectrogram_images.append(spectrogram_image)

    return spectrogram_images

# The filename is stored in the `filename` attribute of the `args` object
filename = args.input

# Generate a list of spectrogram images from the MP3 file
spectrogram_images = spectrogram_images_from_file(
    filename=filename,
    max_volume=args.maxvol,
    power_for_image=args.powerforimage,
    nmels=args.nmels,
    duration=args.duration
    )

os.makedirs(args.output, exist_ok=True)
input_filename_only = Path(args.input).stem
# Iterate over the list of images and save each one to a separate file
print("SAVING SPECTOGRAM IMAGES")
for i, image in enumerate(spectrogram_images):
    # Generate output filename for this image
    output_filename = f"{args.output}/{input_filename_only}_{i:05d}.png"
    image.save(output_filename)

print("FINISHED")
