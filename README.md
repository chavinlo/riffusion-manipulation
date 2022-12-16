# Riffusion Manipulation Tools

# Usage

## Flags
The following arguments/flags are available on all convertors:

`-i / --input INPUTFILE.ext`

`-o / --output OUTPUTFILE.ext`

`-m / --maxvol [integer]` : Maximun volume, 50+ Okay quality, 100+ Good Quality, 255+ Max Quality

`-p / --powerforimage [float]` : Amount of power to use. 0.25-0.30 recommended. Too low will create loud noise, too high will create silent static

`-n / --nmels [integer]` : n_mels to use. Must match the ones on the image. Basically the width. 512 is the default used by the webUI, the higher it is the less compression is used and higher quality. Maximun is somewhere 1280.

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

# More info
The result images are in 1 channel, Black and White. In order for these to be accepted by Stable Diffusion tools, you need to convert them into RGB. The Riffusion inference server also does this. Most webUIs do this by default.

# Experiments

Experiments on variables are available at tests/

Currently the only experiment available is one done with [Planet Girl from ALIEN POP](https://youtu.be/EzSC4PFnYLY?t=19) at 0:19, clip.wav being the original audio, and configurations used being available on the folder names.

# Support
If you need help with this tool, join the SAIL discord and go to the #riffuser channel: https://discord.gg/KfjghC3ppS

You can also join the (unofficial) riffuser discord: https://discord.gg/HWdanyzvRt