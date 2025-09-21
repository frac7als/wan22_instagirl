import subprocess
import os
import modal

# It's good practice to list dependencies in a structured way
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    # Consolidated list of Python dependencies
    .pip_install(
        "gguf",
        "llama-cpp-python",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59"
    )
)

# ## Downloading custom nodes
image = image.run_commands(
    "comfy node install --fast-deps was-node-suite-comfyui@1.0.2",
    "git clone https://github.com/ChenDarYen/ComfyUI-NAG.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-NAG",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/city96/ComfyUI-GGUF.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-GGUF",
    "git clone https://github.com/rgthree/rgthree-comfy.git /root/comfy/ComfyUI/custom_nodes/rgthree-comfy",
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
)

def hf_download():
    from huggingface_hub import hf_hub_download
    import requests
    import shutil

    # Create necessary directories
    diffusion_models_dir = "/root/comfy/ComfyUI/models/diffusion_models"
    vae_dir = "/root/comfy/ComfyUI/models/vae"
    text_encoders_dir = "/root/comfy/ComfyUI/models/text_encoders"
    lora_dir = "/root/comfy/ComfyUI/models/loras"
    
    os.makedirs(diffusion_models_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(text_encoders_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    # ========================================================================
    # WAN DIFFUSION MODELS (SafeTensors format)
    # ========================================================================
    
    # High Noise Model
    wan_model_high_noise = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {wan_model_high_noise} {os.path.join(diffusion_models_dir, 'wan2.2_i2v_high_noise_14B_fp16.safetensors')}",
        shell=True,
        check=True,
    )
    
    # Low Noise Model
    wan_model_low_noise = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {wan_model_low_noise} {os.path.join(diffusion_models_dir, 'wan2.2_i2v_low_noise_14B_fp16.safetensors')}",
        shell=True,
        check=True,
    )

    # ========================================================================
    # VAE MODEL
    # ========================================================================
    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {vae_model} {os.path.join(vae_dir, 'wan_2.1_vae.safetensors')}",
        shell=True,
        check=True,
    )

    # ========================================================================
    # TEXT ENCODER
    # ========================================================================
    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {t5_model} {os.path.join(text_encoders_dir, 'umt5_xxl_fp8_e4m3fn_scaled.safetensors')}",
        shell=True,
        check=True,
    )

    # ========================================================================
    # LORA MODELS
    # ========================================================================
    
    # WAN LightX2V LoRA (for low noise model)
    lightx2v_lora = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {lightx2v_lora} {os.path.join(lora_dir, 'Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors')}",
        shell=True,
        check=True,
    )

    # ========================================================================
    # CIVITAI DOWNLOADS (LoRA Models)
    # ========================================================================
    
    def download_civitai_file(model_version_id, filename, target_dir):
        """Download file from CivitAI using model version ID"""
        target_path = os.path.join(target_dir, filename)
        url = f"https://civitai.com/api/download/models/{model_version_id}"
        print(f"Downloading {filename} from CivitAI...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Downloaded {filename} successfully!")

    # Instagirlv2.5 LoRA (updated version from the image)
    # Model Version ID from the CivitAI page: appears to be the latest version
    try:
        download_civitai_file(
            "1822984",  # This is from the URL in your first image
            "Instagirlv2.5.safetensors",
            lora_dir
        )
    except Exception as e:
        print(f"Failed to download Instagirlv2.5 LoRA: {e}")
    
    # l3n0v0 LoRA (from second image)
    # Model Version ID: 2006914 (from the URL in your second image)
    try:
        download_civitai_file(
            "2006914",
            "l3n0v0.safetensors", 
            lora_dir
        )
    except Exception as e:
        print(f"Failed to download l3n0v0 LoRA: {e}")
    
    # Note: Skipping Instamodel_1 LoRA as requested

    print("All model downloads completed!")

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0", "requests")
    .run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
    )
)

app = modal.App(name="instagirlv23-comfyui", image=image)


@app.function(
    max_containers=1,
    gpu="L40S",
    volumes={"/cache": vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
