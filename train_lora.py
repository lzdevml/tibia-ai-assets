import subprocess, sys, os, io, zipfile, requests

# 1) kohya sd-scripts
if not os.path.exists('/content/sd-scripts'):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/kohya-ss/sd-scripts', '/content/sd-scripts'])
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                'accelerate', 'transformers', 'diffusers', 'safetensors', 'bitsandbytes',
                'xformers', 'opencv-python-headless', 'ftfy', 'einops', 'voluptuous', 'toml', 'huggingface_hub'])

# 2) dataset -> kohya layout (train/<repeats>_tibia/*.png+*.txt)
DST = '/content/train/12_tibia'
os.makedirs(DST, exist_ok=True)
z = requests.get('https://raw.githubusercontent.com/lzdevml/tibia-ai-assets/main/tibia_humanoid_dataset.zip', timeout=120).content
zf = zipfile.ZipFile(io.BytesIO(z))
for n in zf.namelist():
    if n.endswith('.png') or n.endswith('.txt'):
        open(os.path.join(DST, os.path.basename(n)), 'wb').write(zf.read(n))
print('dataset files:', len(os.listdir(DST)))

os.makedirs('/content/lora_out', exist_ok=True)

# 3) train
cmd = [
    'accelerate', 'launch', '--num_processes=1', '--num_machines=1', '--mixed_precision=fp16',
    '/content/sd-scripts/sdxl_train_network.py',
    '--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0',
    '--train_data_dir=/content/train',
    '--resolution=512,512',
    '--output_dir=/content/lora_out', '--output_name=tibia_oblique',
    '--save_model_as=safetensors',
    '--network_module=networks.lora', '--network_dim=32', '--network_alpha=16',
    '--train_batch_size=2', '--max_train_steps=1600', '--learning_rate=1e-4',
    '--unet_lr=1e-4', '--text_encoder_lr=5e-5',
    '--optimizer_type=AdamW8bit', '--lr_scheduler=cosine', '--lr_warmup_steps=100',
    '--mixed_precision=fp16', '--save_precision=fp16',
    '--gradient_checkpointing', '--cache_latents', '--xformers', '--no_half_vae',
    '--save_every_n_steps=400', '--seed=42', '--max_data_loader_n_workers=2',
]
print('starting training...')
subprocess.run(cmd)
print('TRAIN_DONE', os.listdir('/content/lora_out'))
