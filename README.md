# Work-in-progress

<p align="center">
  <img src="">
</p>

<br>
<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML-GPU/refs/heads/main/Other/Configurable.png">
</p>

<br>
<br>

## Hardware

```text
* SSD     $ 43 amazon.com/dp/B09ZYPTXS4
* PSU     $ 45 amazon.com/dp/B0B4MVDRX4
* RAM*2   $ 80 amazon.com/dp/B0143UM4TC
* MB      $ 90 amazon.com/dp/B08KH12V25
* CPU     $160 amazon.com/dp/B091J3NYVF
* GPU     $285 amazon.com/dp/B08WPRMVWB
```

Flashback here

PCIE 3 v 4 grift don't fall for it

4+4: MB  (say "the connector that's secretly split in half if you take a closer look.")

6+2: GPU

grey/black slots only: if 2 RAM stick

<br>
<br>

## Software (IN THIS ORDER!)

* Safe preset overclock: change "Normal" to "Optimal" in BIOS, and update BIOS time (24-hour time.)
* Do a fresh install of Devuan Linux (or Debian-based Linux).
* ```apt install firmware-amd-graphics xserver-xorg-video-amdgpu``` then reboot. (Full res for CPU's iGPU.)
* ```apt install nvidia-driver``` then reboot.
* ```nvidia-smi``` (just to verify) - should show driver version & GPU details.
* ```apt install nvidia-cuda-toolkit```.
* ```nvcc --version``` (just to verify) - should show CUDA version.
* ```apt install python3-torch```.
* ```apt install geany psensor```.
* Grab a zip of this repository and extract it.
* Open ML-GPU.py in Geany (the text editor you just installed.)
* Replace "python" with "python3" in Geany's execute command.
* Open psensor (the temperature display you just installed.)

<br>
<br>

## Play with the given training-data

* Set "retrain" to 20 (just for this.)
* Hit F5 to run.
* Use option 1 (creates a model.)
* Use option 2 (always verify that your model can generalize on the given training-data by scoring ~95%.)
* Replace "training-data", like with that of "github.com/compromise-evident/semiprime-training-data".
* Be sure to adjust "YOUR CONTROLS" as needed!

<br>
<br>

### MB power button pins (short them or attach momentary-switch)

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML-GPU/refs/heads/main/Other/Power_button_pins.png">
</p>

<br>
<br>

## Replace the given training-data (train.txt & test.txt)

```text

   label                       your data
     |                             |


     2 ------@@@-@--------@@@@@@-------------------------@
     9 @--------------------------------------------------------@@
     8 ----------------------------@@
     8 --------------------------@@-------------------------
     0 -------@-@-@-@
     36 --------------@-@@@@-@
     300 --------------------------------------@-@
     5000 ----------@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-------@
     0 --------@-

```

The given training-data is a text version of MNIST, with labels & data combined like above.
**train.txt** contains 60,000 labeled images of handwritten digits
(one label & image per line of text.) An image becomes legible if you arrange
its 784 characters "---@@@---" by stacking rows of 28 (it's a 28x28 image.)
**test.txt** contains 10,000 more such lines not found among the 60k.
See **visual_for_you.txt** to see each image
already stacked 28x28.

So, train.txt & test.txt are the same but of unique items.
Replace them with anything like above.
"-" is normalized as "0.0" (less "seen" by the model)
while "@" is normalized as "1.0" (what the model pays attention to.)
If ```longest``` is set to 900 for example,
then "-" is appended until your data string is 900 long.
It's completely safe. If your string is longer than ```longest```,
only the first ```longest``` characters of that string will be used.
Each line in train.txt & test.txt: ```label```, ```space```, ```- and @```, ```new line``` (\n.)
Each line in cognize.txt: ```- and @```, ```new line``` (\n.)
(You have to create cognize.txt.)
