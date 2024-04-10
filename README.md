# webui_wrap

## Install

```shell
pip install -r requirements.txt
```

## Launch

### Deploy a A1111 WebUI on your environment

Follow this README: https://github.com/AUTOMATIC1111/stable-diffusion-webui

On Ubuntu Environment, one-click installation with `webui.sh` is recommended (
reference: https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file#automatic-installation-on-linux).

1. Install dependencies

```shell
# Debian-based:
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
# Red Hat-based:
sudo dnf install wget git python3 gperftools-libs libglvnd-glx 
# openSUSE-based:
sudo zypper install wget git python3 libtcmalloc4 libglvnd
# Arch-based:
sudo pacman -S wget git python3
```

2. Download and use the auto-installer

```shell
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
chmod +x webui.sh

# Auto-launch webui, all the dependencies will be installed on your first-time launch
# `--xformers`` is strongly recommended, it can significantly speed up the inference (about 70%) and reduce the vram usage (about 40%)
# `--share` and `--no-gradio-queue` is required when you are using proxies, otherwise the websocket will down
./webui.sh -f \
  --port 10188 \
  --api \
  --share --no-gradio-queue \ 
  --max-batch-count 128 \
  --listen \
  --enable-insecure-extension-access \
  --xformers
```

3. After these steps, you can access `https://127.0.0.1:10188`, there should be a webui on that.

ATTENTION: **DONT INSTALL WEBUI IN ANY OTHER EXISTING CONDA/VENV ENVIRONMENTS**. There are many very strict dependencies
limitations in a41 webui, that will break down your existing environment.
The best practice is to install it in a newly created conda environment

```shell
# make your own environment
mkdir -p /dir/for/webui
cd /dir/for/webui
conda create -n webui python=3.10
conda activate webui
conda install cudnn

# download and install
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
chmod +x webui.sh
./webui.sh -f \
  --port 10188 \
  --api \
  --share --no-gradio-queue \ 
  --max-batch-count 128 \
  --listen \
  --enable-insecure-extension-access \
  --xformers
```

Then the working directory will be created at `/dir/for/webui/stable-diffusion-webui`, it is using a venv inside, placed
at `/dir/for/webui/stable-diffusion-webui`. If you want to install something else, just

```shell
cd /dir/for/webui/stable-diffusion-webui
source venv/bin/activate
pip install xxxx
```

### Launch A1111 WebUI with api-only mode

It's a really annoying thing, that a41 webui cannot support UI-mode and API-mode at the same time.
That means when you need to use API for integration with other applications, you will not be able to use webui for human
usage.

Launch with api-only mode with `--nowebui`:

```shell
./webui.sh -f \
  --port 10188 \
  --api \
  --share --no-gradio-queue \ 
  --max-batch-count 128 \
  --listen \
  --enable-insecure-extension-access \
  --xformers \
  --nowebui
```

### Launch Webui WRAP

Please note that, you should launch the webui with api-only mode before launching webui wrap.

E.g. launched at `http://10.140.1.178:33088`, and them

```shell
# set environment variable
export CH_WEBUI_SERVER=http://10.140.1.178:33088

python app.py --bind_all --share --port 10187
```

The webui wrap UI will be launched at `http://127.0.0.1:10187`

### Adding Base Model

```shell
cd /dir/for/webui/stable-diffusion-webui/models/Stable-diffusion

# download nai anime base model
curl -o 'nai.ckpt' -L 'https://huggingface.co/deepghs/animefull-latest-ckpt/resolve/main/model.ckpt'
```

And then just refresh the base model list on your webui page.

### Adding Lora Model

```shell
cd /dir/for/webui/stable-diffusion-webui/models/Lora

# download paimon lora
# for more information of this lora, see: https://huggingface.co/CyberHarem/paimon_genshin
curl -O -L 'https://huggingface.co/CyberHarem/paimon_genshin/resolve/main/3816/paimon_genshin.safetensors'
```

And then just refresh the lora list on your webui page.

