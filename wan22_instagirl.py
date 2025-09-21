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
    import zipfile
    import tempfile
    import re

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
    # ========================================================================
    def hf_try_download(repo_id, filename, target_basename):
        try:
            p = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir="/cache")
            target_path = os.path.join(unet_dir, target_basename)
            subprocess.run(f"ln -sf {p} {target_path}", shell=True, check=True)
            subprocess.run(f"cp -f {p} {target_path}.copy", shell=True, check=True)
            print(f"✔ Downloaded {target_basename} from {repo_id}/{filename}")
            return True
        except Exception as e:
            print(f"✖ Fallback: {repo_id}/{filename} not available ({e})")
            return False

    # LowNoise GGUF
    if not (
        hf_try_download(
            "QuantStack/Wan2.2-T2V-A14B-GGUF",
            "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
        )
        or hf_try_download(
            "Phr00t/WAN2.2-14B-Rapid-AllInOne",
            "Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
        )
    ):
        print("!! Could not fetch LowNoise GGUF.")

    # HighNoise GGUF
    if not (
        hf_try_download(
            "QuantStack/Wan2.2-T2V-A14B-GGUF",
            "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
        )
        or hf_try_download(
            "Phr00t/WAN2.2-14B-Rapid-AllInOne",
            "Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
            "Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
        )
    ):
        print("!! Could not fetch HighNoise GGUF.")

    # ========================================================================
    # VAE & TEXT ENCODER
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
    # WAN LightX2V LoRA
    # ========================================================================
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
    # Instagirlv2.5 ZIP handling
    # ========================================================================
    def download_civitai_zip(model_version_id, filename, target_dir):
        target_path = os.path.join(target_dir, filename)
        url = f"https://civitai.com/api/download/models/{model_version_id}"
        print(f"Downloading ZIP {filename} from CivitAI...")
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, stream=True, headers=headers)
        r.raise_for_status()
        with open(target_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        print(f"Downloaded ZIP: {target_path}")
        return target_path

    def process_lora_zip(zip_path, target_dir, base_alias="Instagirlv2.5"):
        import zipfile, tempfile, re, shutil

        print(f"Extracting: {zip_path}")
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(td)

            # Find .safetensors inside
            found = []
            for root, _, files in os.walk(td):
                for fn in files:
                    if fn.lower().endswith(".safetensors"):
                        found.append(os.path.join(root, fn))

            if not found:
                raise RuntimeError("No .safetensors found in ZIP")

            os.makedirs(target_dir, exist_ok=True)
            copied = []
            for src in found:
                dst = os.path.join(target_dir, os.path.basename(src))
                shutil.copyfile(src, dst)
                copied.append(dst)
                print(f"→ Copied {dst}")

        # Detect high/low
        high_pat = re.compile(r"high", re.IGNORECASE)
        low_pat = re.compile(r"low", re.IGNORECASE)

        high_file = next((p for p in copied if high_pat.search(os.path.basename(p))), None)
        low_file = next((p for p in copied if low_pat.search(os.path.basename(p))), None)

        if not (high_file and low_file) and len(copied) == 2:
            a, b = sorted(copied)
            if not high_file: high_file = a
            if not low_file:  low_file = b

        HIGH_TARGET = os.path.join(target_dir, f"{base_alias}-HIGH.safetensors")
        LOW_TARGET  = os.path.join(target_dir, f"{base_alias}-LOW.safetensors")

        def write_canonical(src, dst, label):
            if not src:
                print(f"!! No {label} file found")
                return
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o644)
            print(f"✓ Wrote {label}: {dst}")

        write_canonical(high_file, HIGH_TARGET, "HIGH")
        write_canonical(low_file, LOW_TARGET, "LOW")

        try: os.remove(zip_path)
        except: pass

    try:
        zip_path = download_civitai_zip("2180477", "Instagirlv2.5.zip", lora_dir)
        process_lora_zip(zip_path, lora_dir)
    except Exception as e:
        print(f"Failed Instagirlv2.5: {e}")

    print("All model downloads completed!")

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0", "requests")
    .run_function(
        hf_download,
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
    os.environ["COMFYUI_MODEL_DIR"] = "/root/comfy/ComfyUI/models"

    print("\n=== LORA INVENTORY ===")
    subprocess.run("ls -la /root/comfy/ComfyUI/models/loras || true", shell=True, check=False)
    subprocess.run("sha256sum /root/comfy/ComfyUI/models/loras/Instagirlv2.5-*.safetensors || true", shell=True, check=False)

    print("\n=== UNET INVENTORY ===")
    subprocess.run("ls -la /root/comfy/ComfyUI/models/unet || true", shell=True, check=False)

    print("\n=== GGUF NODE PRESENCE ===")
    subprocess.run("ls -la /root/comfy/ComfyUI/custom_nodes/ComfyUI-GGUF || true", shell=True, check=False)

    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
