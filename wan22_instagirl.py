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
    # Install comfy-image-saver with dependencies
    "git clone https://github.com/giriss/comfy-image-saver.git /root/comfy/ComfyUI/custom_nodes/comfy-image-saver",
    "cd /root/comfy/ComfyUI/custom_nodes/comfy-image-saver && pip install -r requirements.txt",
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
    unet_dir = "/root/comfy/ComfyUI/models/unet"  # for GGUF models used by UnetLoaderGGUF

    os.makedirs(diffusion_models_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(text_encoders_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    os.makedirs(unet_dir, exist_ok=True)

    # ========================================================================
    # WAN UNET (GGUF) MODELS — for ComfyUI-GGUF's UnetLoaderGGUF
    # Replaces prior safetensors diffusion model downloads.
    # Filenames match the workflow exactly.
    # ========================================================================
    def hf_try_download(repo_id, filename, target_basename):
        try:
            p = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir="/cache")
            subprocess.run(
                f"ln -sf {p} {os.path.join(unet_dir, target_basename)}",
                shell=True,
                check=True,
            )
            print(f"✔ Downloaded {target_basename} from {repo_id}/{filename}")
            return True
        except Exception as e:
            print(f"✖ Fallback: {repo_id}/{filename} not available ({e})")
            return False

    # LowNoise GGUF (exact name used by the workflow)
    if not (
        hf_try_download(
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "split_files/gguf/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
        )
        or hf_try_download(
            "Phr00t/WAN2.2-14B-Rapid-AllInOne",
            "Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
        )
    ):
        print("!! Could not fetch LowNoise GGUF. Please provide an alternate repo/path.")

    # HighNoise GGUF (exact name used by the workflow)
    if not (
        hf_try_download(
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "split_files/gguf/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
        )
        or hf_try_download(
            "Phr00t/WAN2.2-14B-Rapid-AllInOne",
            "Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
        )
    ):
        print("!! Could not fetch HighNoise GGUF. Please provide an alternate repo/path.")

    # ========================================================================
    # VAE MODEL (safetensors as required by your workflow)
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
    # TEXT ENCODER (safetensors as required by your workflow)
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()

        import shutil as _shutil
        with open(target_path, "wb") as f:
            _shutil.copyfileobj(response.raw, f)
        print(f"Downloaded {filename} successfully!")

    # Instagirlv2.5 LoRA (modelVersionId=2180477)
    try:
        download_civitai_file(
            "2180477",
            "Instagirlv2.5.safetensors",
            lora_dir,
        )
    except Exception as e:
        print(f"Failed to download Instagirlv2.5 LoRA: {e}")
        print("Manual download available at: https://civitai.com/models/1822984?modelVersionId=2180477")

    # l3n0v0 LoRA (modelVersionId=2006914)
    try:
        download_civitai_file(
            "2006914",
            "l3n0v0.safetensors",
            lora_dir,
        )
    except Exception as e:
        print(f"Failed to download l3n0v0 LoRA: {e}")

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
