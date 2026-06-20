import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '-U', 'diffusers', 'transformers', 'accelerate', 'peft', 'safetensors'])
import torch, requests, io, json
import numpy as np
from PIL import Image
try:
    import cv2
except Exception:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'opencv-python-headless']); import cv2
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

KEY = '4f95e1b95c22a001687576e3c07fcd33'
H = {'Authorization': 'Bearer ' + KEY, 'User-Agent': 'Mozilla/5.0'}

def getb(i):
    u = 'https://orchestration-new.civitai.com/v2/consumer/blobs/' + i + '.png'
    return Image.open(io.BytesIO(requests.get(u, headers=H, timeout=120).content)).convert('RGB')

if 'cnpipe' not in globals():
    cn = ControlNetModel.from_pretrained('diffusers/controlnet-canny-sdxl-1.0', torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
    cnpipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', controlnet=cn, vae=vae,
        torch_dtype=torch.float16, variant='fp16')
    cnpipe.load_lora_weights('nerijs/pixel-art-xl', weight_name='pixel-art-xl.safetensors')
    cnpipe.fuse_lora(lora_scale=1.2)
    cnpipe.to('cuda'); cnpipe.enable_vae_tiling()
    print('controlnet pipe ready')

M = requests.get('https://raw.githubusercontent.com/lzdevml/tibia-ai-assets/main/reaper_run.json', timeout=60).json()

def canny_from(bid):
    base = getb(bid).resize((1024, 1024), Image.NEAREST)
    a = np.array(base)
    e = cv2.Canny(a, 60, 140)
    return Image.fromarray(np.stack([e, e, e], -1))

PROMPT = 'pixel art, a grim reaper, hooded figure wearing a dark purple tattered robe, holding a scythe, glowing violet eyes, top-down rpg game character sprite'
NEG = 'blurry, 3d render, photo, realistic, smooth shading, jpeg artifacts'

def gen(bid, cscale, seed=777):
    ctrl = canny_from(bid)
    g = torch.Generator(device='cuda').manual_seed(seed)
    out = cnpipe(prompt=PROMPT, negative_prompt=NEG, image=ctrl,
                 num_inference_steps=30, guidance_scale=7.5,
                 controlnet_conditioning_scale=cscale, generator=g).images[0]
    return ctrl, out

def up(name, im):
    b = io.BytesIO(); im.save(b, 'PNG')
    r = requests.post('https://orchestration.civitai.com/v2/consumer/blobs',
                      headers={'Authorization': 'Bearer ' + KEY, 'User-Agent': 'Mozilla/5.0', 'Content-Type': 'image/png'},
                      data=b.getvalue())
    return r.json().get('id')

out = {}
for cs in [0.6, 0.9]:
    ctrl, res = gen(M['frames']['S_p0'], cs)
    small = res.resize((64, 64), Image.NEAREST).resize((512, 512), Image.NEAREST)
    out[f'cn_S_full_{cs}'] = up(f'full_{cs}', res)
    out[f'cn_S_px_{cs}'] = up(f'px_{cs}', small)
    if cs == 0.6:
        out['cn_S_ctrl'] = up('ctrl', ctrl.resize((512, 512), Image.NEAREST))
    print('done cs', cs)
print('CNJSON', json.dumps(out))
