"""
ComfyUI-GPT-Image2 (v2.2 Stable)
Refactorizado para máxima resiliencia ("Ultra-Defensivo").
Evita parámetros no soportados y dependencias de serialización estrictas.
[ignoring loop detection]
"""

import base64
import math
import time
import json
import asyncio
from io import BytesIO

try:
    from comfy.model_management import InterruptProcessingException as _ComfyInterrupt
except ImportError:
    _ComfyInterrupt = None

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_bytesio,
    downscale_image_tensor,
    sync_op_raw,
    validate_string,
)
from comfy_api_nodes.nodes_openai import (
    calculate_tokens_price_image_1,
    calculate_tokens_price_image_1_5,
    calculate_tokens_price_image_2_0,
)

# ─── Configuración de Tamaños ────────────────────────────────────────

def _ratio_label(w: int, h: int) -> str:
    g = math.gcd(w, h)
    rw, rh = w // g, h // g
    mp = (w * h) / 1_000_000
    return f"{rw}:{rh}  {mp:.1f}MP"

SIZE_OPTIONS = [
    ("auto",                                          "auto"),
    (f"1024 × 1024   ({_ratio_label(1024, 1024)})",   "1024x1024"),
    (f"1024 × 1536   ({_ratio_label(1024, 1536)})",   "1024x1536"),
    (f"1536 × 1024   ({_ratio_label(1536, 1024)})",   "1536x1024"),
    (f"2048 × 2048   ({_ratio_label(2048, 2048)})",   "2048x2048"),
    (f"2048 × 1152   ({_ratio_label(2048, 1152)})",   "2048x1152"),
    (f"1152 × 2048   ({_ratio_label(1152, 2048)})",   "1152x2048"),
    (f"3840 × 2160   ({_ratio_label(3840, 2160)})",   "3840x2160"),
    (f"2160 × 3840   ({_ratio_label(2160, 3840)})",   "2160x3840"),
]

DISPLAY_TO_API = {label: val for label, val in SIZE_OPTIONS}
DISPLAY_LABELS = [label for label, _ in SIZE_OPTIONS]
API_TO_DISPLAY = {val: label for label, val in SIZE_OPTIONS if val != "auto"}

# ─── Logs de Consola Premium ─────────────────────────────────────────

NODE_TAG = "GPTImage2"

def _log_header(model: str, api_size: str, n: int, has_image: bool):
    mode = "Edit" if has_image else "Generate"
    size_display = API_TO_DISPLAY.get(api_size, api_size)
    print(f"\n{'═' * 60}")
    print(f"🎨 {NODE_TAG} | {mode} {'× ' + str(n) if n > 1 else ''}")
    print(f"{'─' * 60}")
    print(f"   Model:   {model}")
    print(f"   Size:    {size_display}")
    print(f"{'─' * 60}")

def _log_success(count: int, w: int, h: int, elapsed: float):
    print(f"   ✅ Done   | {count} image(s) received at {w}×{h} ({elapsed:.1f}s)")
    print(f"{'═' * 60}\n")

# ─── Decodificador de Respuesta ──────────────────────────────────────

async def _validate_response_raw(resp_dict, timeout=None) -> torch.Tensor:
    """Parsea el JSON de respuesta manualmente para evitar fallos de Pydantic."""
    data_list = resp_dict.get("data", [])
    if not data_list:
        error = resp_dict.get("error", {})
        msg = error.get("message", "Unknown API error")
        raise ValueError(f"API Error: {msg}")
    
    tensors = []
    for item in data_list:
        b64 = item.get("b64_json")
        url = item.get("url")
        
        if b64:
            img_io = BytesIO(base64.b64decode(b64))
        elif url:
            img_io = BytesIO()
            await download_url_to_bytesio(url, img_io, timeout=timeout)
        else:
            continue
        
        pil_img = Image.open(img_io).convert("RGBA")
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    
    if not tensors:
        raise ValueError("No valid image data found in response")
        
    return torch.stack(tensors, dim=0)

# ─── Nodo Principal ──────────────────────────────────────────────────

class GPTImage2Node(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GPTImage2Node",
            display_name="GPT Image 2 🎨",
            category="api node/image/OpenAI",
            description="Decorated size labels with aspect ratio and megapixels. Ultra-defensive implementation.",
            inputs=[
                IO.Image.Input("image", optional=True),
                IO.Mask.Input("mask", optional=True),
                IO.String.Input("prompt", default="", multiline=True),
                IO.Combo.Input("size", default=DISPLAY_LABELS[1], options=DISPLAY_LABELS),
                IO.Int.Input("n", default=1, min=1, max=8),
            ],
            outputs=[IO.Image.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
        )

    @classmethod
    async def execute(cls, prompt: str, image: Input.Image | None = None, mask: Input.Image | None = None,
                     n: int = 1, size: str = DISPLAY_LABELS[1]) -> IO.NodeOutput:
        
        model = "gpt-image-2"
        t0 = time.perf_counter()
        validate_string(prompt, strip_whitespace=False)
        api_size = DISPLAY_TO_API.get(size, size)

        # Selección de extractor de precio
        price_fn = {
            "gpt-image-1": calculate_tokens_price_image_1,
            "gpt-image-1.5": calculate_tokens_price_image_1_5,
            "gpt-image-2": calculate_tokens_price_image_2_0
        }.get(model, calculate_tokens_price_image_2_0)

        _log_header(model, api_size, n, image is not None)

        # Preparación de archivos para Edición (Caching bytes)
        image_bytes_cache = []
        mask_bytes_cache = None
        batch_count = 1

        if image is not None:
            batch_count = image.shape[0]
            for i in range(batch_count):
                scaled = downscale_image_tensor(image[i:i+1], total_pixels=2048*2048).squeeze()
                pil = Image.fromarray((scaled.numpy() * 255).astype(np.uint8))
                buf = BytesIO()
                pil.save(buf, format="PNG")
                image_bytes_cache.append(buf.getvalue())
            
            if mask is not None:
                if batch_count > 1:
                    print(f"   ⚠️  Warning: Mask provided with batch input. Only supported for single image. Skipping mask.")
                else:
                    h, w = mask.shape[1:]
                    rgba_mask = torch.zeros(h, w, 4)
                    rgba_mask[:, :, 3] = 1 - mask.squeeze()
                    scaled_m = downscale_image_tensor(rgba_mask.unsqueeze(0), total_pixels=2048*2048).squeeze()
                    m_buf = BytesIO()
                    Image.fromarray((scaled_m.numpy() * 255).astype(np.uint8)).save(m_buf, format="PNG")
                    mask_bytes_cache = m_buf.getvalue()

        # Ejecución
        iterations = max(1, n // batch_count) if image is not None else 1
        n_param = 1 if image is not None else n

        async def _make_request(idx):
            files = None
            if image is not None:
                files = []
                for j, img_bytes in enumerate(image_bytes_cache):
                    field_name = "image" if batch_count == 1 else "image[]"
                    files.append((field_name, (f"img_{j}.png", BytesIO(img_bytes), "image/png")))
                if mask_bytes_cache is not None:
                    files.append(("mask", ("mask.png", BytesIO(mask_bytes_cache), "image/png")))

            req_data = {
                "model": str(model),
                "prompt": str(prompt),
                "n": n_param,
            }
            if api_size and api_size != "auto":
                req_data["size"] = api_size

            path = "/proxy/openai/images/edits" if files else "/proxy/openai/images/generations"
            
            safe_payload = {k: (v if k != "prompt" else (v[:40] + "..." if len(v) > 40 else v)) for k, v in req_data.items()}
            msg = f"   📡 [{idx+1}/{iterations}] Sending to {path}" if iterations > 1 else f"   📡 Sending to {path}"
            print(f"{msg}: {json.dumps(safe_payload)}")

            raw_resp = await sync_op_raw(
                cls, ApiEndpoint(path=path, method="POST"),
                data=req_data,
                files=files,
                content_type="multipart/form-data" if files else "application/json",
                price_extractor=price_fn,
                timeout=300.0 # 5 min timeout
            )
            
            tensor = await _validate_response_raw(raw_resp)
            _, h_res, w_res, _ = tensor.shape
            print(f"   🖼️  Received {tensor.shape[0]} image(s) at {w_res}x{h_res} from req [{idx+1}]")
            return tensor

        try:
            tasks = [_make_request(i) for i in range(iterations)]
            result_tensors = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise  # Cancelación silenciosa
        except Exception as e:
            # Cancelación por el usuario → salir sin ruido
            if _ComfyInterrupt and isinstance(e, _ComfyInterrupt):
                raise
            if "InterruptProcessingException" in type(e).__name__:
                raise
            err_msg = str(e)
            print(f"   ❌ Error: {err_msg}")
            if "Unknown parameter" in err_msg:
                print(f"   💡 Tip: Your proxy might be too old or strict.")
            raise

        # Concatenación final con validación de dimensiones
        if not result_tensors:
            raise RuntimeError("No images were returned from the API.")

        try:
            # Validar que todos los tensores tengan el mismo H y W antes de concatenar
            first_shape = result_tensors[0].shape[1:3]
            for idx, t in enumerate(result_tensors[1:], 1):
                if t.shape[1:3] != first_shape:
                    print(f"   ⚠️  Dimension mismatch in batch! Batch 0 is {first_shape}, Batch {idx} is {t.shape[1:3]}")
                    print(f"   💡 Resizing Batch {idx} to match Batch 0...")
                    # Redimensionar usando interpolación básica si hay desajuste
                    t_pil = [Image.fromarray((img.numpy() * 255).astype(np.uint8)) for img in t]
                    t_resized = []
                    for img in t_pil:
                        img_res = img.resize((first_shape[1], first_shape[0]), Image.Resampling.LANCZOS)
                        t_resized.append(torch.from_numpy(np.asarray(img_res).astype(np.float32) / 255.0))
                    result_tensors[idx] = torch.stack(t_resized, dim=0)

            res = torch.cat(result_tensors, dim=0)
            _log_success(res.shape[0], res.shape[2], res.shape[1], time.perf_counter() - t0)
            return IO.NodeOutput(res)
        except Exception as e:
            raise RuntimeError(f"Failed to assemble image batch: {e}")

class GPTImage2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [GPTImage2Node]

async def comfy_entrypoint() -> GPTImage2Extension:
    return GPTImage2Extension()
