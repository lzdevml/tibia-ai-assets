import torch, requests, io, json
from PIL import Image
H = {'Authorization': 'Bearer 4f95e1b95c22a001687576e3c07fcd33', 'User-Agent': 'Mozilla/5.0'}
M = requests.get('https://raw.githubusercontent.com/lzdevml/tibia-ai-assets/main/reaper_run.json', timeout=60).json()
def gb(i):
    u = 'https://orchestration-new.civitai.com/v2/consumer/blobs/' + i + '.png'
    return Image.open(io.BytesIO(requests.get(u, headers=H, timeout=120).content)).convert('RGB').resize((1024, 1024))
master = gb(M['master'])
bases = {k: gb(v) for k, v in M['frames'].items()}
print('inputs', len(bases))
RP = ('a hooded void reaper, grim reaper, tattered dark purple hooded robe, holding a long curved scythe, '
      'glowing violet eyes, dark fantasy game character, top-down rpg character sprite, plain black background')
res = {}
for n, b in bases.items():
    g = torch.Generator(device='cuda').manual_seed(777)
    res[n] = pipe(prompt=RP, negative_prompt=NEG, image=b, ip_adapter_image=master,
                  strength=0.6, num_inference_steps=STEPS, generator=g).images[0]
    print('gen', n)
out = {}
for n, im in res.items():
    bb = io.BytesIO(); im.save(bb, 'PNG'); bb.seek(0)
    r = requests.post('https://catbox.moe/user/api.php', data={'reqtype': 'fileupload'},
                      files={'fileToUpload': (n + '.png', bb, 'image/png')}, timeout=180)
    out[n] = r.text.strip(); print('URL', n, out[n])
print('JSON_RESULT', json.dumps(out))
