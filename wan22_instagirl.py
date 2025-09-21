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
    # Instagirl v2.5 – force the exact Diffusers ZIP link you showed
    # ========================================================================
    def is_zip_file(path):
        try:
            with open(path, "rb") as f:
                sig = f.read(4)
            return sig == b"PK\x03\x04"
        except Exception:
            return False

    def download_url_to(path, url, params=None, chunk_mb=8):
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, headers=headers, params=params or {}, allow_redirects=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return path, r.headers.get("Content-Type", ""), r.headers.get("Content-Disposition", "")

    def ensure_instagirl_high_low_from_files(files, target_dir, base_alias="Instagirlv2.5"):
        """
        Given a list of .safetensors file paths, write/overwrite:
          Instagirlv2.5-HIGH.safetensors
          Instagirlv2.5-LOW.safetensors
        Heuristics choose which is HIGH/LOW; if only one file, duplicate to both.
        """
        import re, shutil
        os.makedirs(target_dir, exist_ok=True)
        high_pat = re.compile(r"(?:^|[-_ ()])high(?:[-_ ()]?noise)?(?:[-_. ()]|$)", re.IGNORECASE)
        low_pat  = re.compile(r"(?:^|[-_ ()])low(?:[-_ ()]?noise)?(?:[-_. ()]|$)",  re.IGNORECASE)

        high_file = next((p for p in files if high_pat.search(os.path.basename(p))), None)
        low_file  = next((p for p in files if  low_pat.search(os.path.basename(p))),  None)

        if not (high_file and low_file) and len(files) == 2:
            a, b = sorted(files)
            if not high_file: high_file = a
            if not low_file:  low_file  = b

        if not files:
            raise RuntimeError("No safetensors files available for HIGH/LOW")

        if len(files) == 1:
            high_file = low_file = files[0]

        HIGH_TARGET = os.path.join(target_dir, f"{base_alias}-HIGH.safetensors")
        LOW_TARGET  = os.path.join(target_dir, f"{base_alias}-LOW.safetensors")

        def write_copy(src, dst, label):
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o644)
            print(f"✓ Wrote {label}: {dst} (from {os.path.basename(src)})")

        write_copy(high_file, HIGH_TARGET, "HIGH")
        write_copy(low_file,  LOW_TARGET,  "LOW")

    def download_and_place_instagirl_pair_from_exact_zip(url, target_dir, base_alias="Instagirlv2.5"):
        """
        Downloads EXACT URL you provided (Diffusers zip), extracts, and ensures
        Instagirlv2.5-HIGH/LOW.safetensors appear in target_dir.
        If the server returns a single file instead of a zip, handle gracefully.
        """
        zip_path = os.path.join(target_dir, f"{base_alias}.zip")
        # Your exact link uses these query params:
        params = {"type": "Model", "format": "Diffusers"}
        path, ctype, cd = download_url_to(zip_path, url, params=params, chunk_mb=16)
        print(f"Downloaded Instagirl ZIP candidate: {path} (ctype={ctype}, cd={cd})")

        safes = []
        if is_zip_file(path):
            print("→ Verified ZIP signature, extracting…")
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(td)
                for root, _, files in os.walk(td):
                    for fn in files:
                        if fn.lower().endswith(".safetensors"):
                            src = os.path.join(root, fn)
                            dst = os.path.join(target_dir, fn)
                            shutil.copyfile(src, dst)
                            safes.append(dst)
                            print(f"→ Extracted {dst}")
            try:
                os.remove(path)
            except Exception:
                pass
        else:
            print("! File is not a ZIP; treating as a single safetensors payload")
            # If not a zip, ensure it's named *.safetensors; if not, append extension.
            final_name = "Instagirlv2.5.safetensors"
            if not path.lower().endswith(".safetensors"):
                new_path = os.path.join(target_dir, final_name)
                shutil.move(path, new_path)
                path = new_path
            safes.append(path)
            print(f"→ Saved single safetensors: {path}")

        ensure_instagirl_high_low_from_files(safes, target_dir, base_alias=base_alias)

    # Use your exact link
    try:
        download_and_place_instagirl_pair_from_exact_zip(
            url="https://civitai.com/api/download/models/2180477",
            target_dir=lora_dir,
            base_alias="Instagirlv2.5",
        )
    except Exception as e:
        print(f"Failed Instagirlv2.5 handling: {e}")

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
