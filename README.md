# Riffusion Manipulation Tools

# Usage
## Convert Audio to Image
To convert an audio into an image use file2img.py:

`python3 file2img.py -f INPUT_AUDIO.wav -o OUTPUT_IMAGE.png`

Note that, this will only convert the last 5 seconds.

For example, to convert invader_by_snailshouse.wav (Credits: INVADER, Snail's House)

`python3 file2img.py -f invader_by_snailshouse.wav -o invader.png`

Will generate the following image:

<img src="invader.png" alt="Spectogram of INVADER" width="512">

This image can be used as a seed on the riffusion webUI.

## Verify / Convert Image to Audio
It is highly recommended to verify that the audio has been correctly converted. You can do soo by using img2audio.py:

`python3 img2audio.py -f invader.png -o invader_rebuild.wav`

This audio is also included in the repository.