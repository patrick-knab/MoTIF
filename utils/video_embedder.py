from __future__ import annotations

import os
import random
import numpy as np
import torch
import clip
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
from typing import List, Tuple, Optional
import cv2
from PIL import Image
import tqdm


def init_repro(seed: int = 42, deterministic: bool = True):
    """Call this at the very top of your notebook/script BEFORE creating any model/processor/device context."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":16:8"  # deterministic cuBLAS on Ampere+, nice default
    )
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism knobs (do this before any CUDA ops)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # older torch may not support signature
            torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Reduce threading nondeterminism
    torch.set_num_threads(1)

    return seed


def _cpu_tensor_or_none(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


class _PickleBackendsMixin:
    def attach_backends(
        self, *, model=None, tokenizer=None, clip_model=None, device=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if getattr(self, "model", None) is not None:
            self.model = self.model.to(self.device).eval()

    def __getstate__(self):
        s = self.__dict__.copy()
        # drop unpicklables
        for k in ("model", "tokenizer", "clip_model", "device"):
            s.pop(k, None)
        # ensure tensors are CPU-picklable
        for k in ("video_embeddings", "text_embeddings"):
            if k in s and s[k] is not None:
                if isinstance(s[k], dict):
                    s[k] = {kk: _cpu_tensor_or_none(vv) for kk, vv in s[k].items()}
                else:
                    s[k] = _cpu_tensor_or_none(s[k])
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        # backends are reattached by caller after unpickle
        self.model = None
        self.tokenizer = None
        self.clip_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoEmbedder(_PickleBackendsMixin):
    def __init__(self, model_name, model, tokenizer, clip_model=None,
                 pe_video_batch_size: Optional[int] = None, pe_target_T: Optional[int] = None):
        self.model_name = model_name.lower()
        self.model = model
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.pe_video_batch_size = pe_video_batch_size
        self.pe_target_T = pe_target_T

        self.dataset_name: Optional[str] = None
        self.video_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.labels: Optional[List[str]] = None
        self.video_window_spans: Dict[str, List[Tuple[float, float]]] = {}
        self.video_meta: Dict[str, Dict[str, float]] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        torch.backends.cudnn.benchmark = True
        self.embed_dim = self._detect_embed_dim()

    # ------------------------- util
    def _detect_embed_dim(self) -> int:
        with torch.inference_mode():
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.model_name == "res50":
                t = self.tokenizer(Image.fromarray(dummy)).unsqueeze(0).to(self.device)
                d = self.model.encode_image(t).shape[-1]
            elif self.model_name in {"clip", "siglip", "siglipl14", "siglip2"}:
                batch = self.tokenizer(images=dummy, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                d = self.model.get_image_features(**batch).shape[-1]
            elif self.model_name == "clip4clip":
                preprocess = Compose(
                    [
                        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
                t = preprocess(Image.fromarray(dummy)).unsqueeze(0).to(self.device)
                d = self.model(t)["image_embeds"].shape[-1]
            elif self.model_name in {"pe-l14", "pe-g14"}:
                # Create a dummy clip of length 16 by repeating a single frame
                # Get image size from model if available, otherwise use defaults
                img_size = getattr(self.model, 'image_size', 336 if self.model_name == "pe-l14" else 448)
                dummy_img = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
                frame = self.tokenizer(dummy_img)
                clip = torch.stack([frame for _ in range(16)], dim=0)  # (T,C,H,W)
                clip = clip.unsqueeze(0).to(self.device)  # (B,T,C,H,W)
                d = self.model.encode_video(clip).shape[-1]
            else:
                raise ValueError(f"Unknown model_name {self.model_name}")
        return int(d)

    def _preprocess_video_pe(
            self,
            video: List[Image.Image],  # now expects a list of PIL Images
            num_frames: int = 4,
            transform: Optional[Compose] = None,
            return_first_frame_for_demo: bool = False
        ) -> Tuple[torch.Tensor, Optional[Image.Image]]:
        total_frames = len(video)
        # Uniformly sample frame indices
        frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
        frames = [video[i] for i in frame_indices]
        # Preprocess frames
        preprocessed_frames = [transform(frame) for frame in frames]

        first_frame = None
        if return_first_frame_for_demo:
            first_frame = frames[0]
        return torch.stack(preprocessed_frames, dim=0), first_frame
    @staticmethod
    def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _sample_indices(n: int, k: int, random: bool) -> List[int]:
        if n <= 0 or k <= 0:
            return []
        if random:
            k = min(k, n)
            return np.random.choice(n, size=k, replace=False).tolist()
        if k >= n:
            return list(range(n))
        step = (n - 1) / (k - 1) if k > 1 else 1e9
        return [int(round(i * step)) for i in range(k)]

    # ------------------------- encoders
    def _encode_images_hf(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """HF CLIP, SigLIP"""
        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        images_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        batch = self.tokenizer(images=images_rgb, return_tensors="pt")
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device.type == "cuda"),
        ):
            feats = self.model.get_image_features(**batch)
        return feats.float().detach().cpu()

    def _encode_images_pe(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """Encode frames using PE-L/14 video model.

        This implementation preserves alignment: it returns one embedding per
        input frame by forming a temporal clip centered on each anchor frame
        (padding at the edges). Clips are length T (default 16) and are
        batched efficiently on GPU.
        """

        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        # Config
        clip_len = 16  # temporal length expected by the video model
        half_clip = clip_len // 2
        max_gpu_batch = 8  # number of clips to encode per forward pass

        # Preprocess all frames once (C,H,W tensors on CPU)
        pil_imgs = [self._bgr_to_pil(f) for f in frames_bgr]
        frames_tensor = [self.tokenizer(img) for img in pil_imgs]
        n = len(frames_tensor)

        def build_clip_around(idx: int) -> torch.Tensor:
            """Return a tensor of shape (T, C, H, W) for anchor frame idx."""
            start = idx - half_clip
            end = start + clip_len
            # Clamp and pad by edge repetition
            frames = []
            for t in range(start, end):
                clamped = min(max(t, 0), n - 1)
                frames.append(frames_tensor[clamped])
            return torch.stack(frames, dim=0)

        embs = []
        with torch.inference_mode():
            # Iterate in micro-batches of clips to control memory
            for s in range(0, n, max_gpu_batch):
                batch_indices = list(range(s, min(s + max_gpu_batch, n)))
                clips = [build_clip_around(i) for i in batch_indices]
                x = torch.stack(clips, dim=0).to(self.device, non_blocking=True)
                # x: (B, T, C, H, W)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(self.device.type == "cuda"),
                ):
                    feats = self.model.encode_video(x)
                embs.append(feats.detach().cpu())

        return torch.cat(embs, dim=0).float()

    def _encode_images_openai_clip(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """OpenAI CLIP RN50."""
        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        pil_imgs = [self._bgr_to_pil(f) for f in frames_bgr]
        x = torch.stack([self.tokenizer(img) for img in pil_imgs], dim=0).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device.type == "cuda"),
        ):
            feats = self.model.encode_image(x)
        return feats.float().detach().cpu()

    def _encode_images_clip4clip(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """CLIP4Clip (expects raw pixel tensors normalized manually)."""
        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        preprocess = Compose(
            [
                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        pil_imgs = [self._bgr_to_pil(f) for f in frames_bgr]
        x = torch.stack([preprocess(img) for img in pil_imgs], dim=0).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode():
            out = self.model(x)["image_embeds"]
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return out.float().detach().cpu()

    # ------------------------- video reading
    def _read_windows(self, video_path: str, window_size: int):
        windows, spans = [], []

        if video_path.lower().endswith(".mp4"):
            # ---- read video ----
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

            if window_size > frame_count:
                frame_count = window_size

            frames = []
            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

        else:
            # ---- read raw images in directory ----
            img_files = sorted(
                [
                    f
                    for f in os.listdir(video_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            frames = [cv2.imread(os.path.join(video_path, f)) for f in img_files]
            frames = [f for f in frames if f is not None]

            frame_count = len(frames)
            fps = 12.0  # fallback since no video container

        # ---- make windows ----
        for i in range(0, frame_count, window_size):
            chunk = frames[i : i + window_size]
            if len(chunk) == 0:
                continue
            windows.append(chunk)
            start_t = i / fps
            end_t = (i + len(chunk) - 1) / fps
            spans.append((start_t, end_t))

        return windows, spans, fps, frame_count

    def _encode_windows(
        self, windows, frames_per_window, random, batch_size
    ) -> np.ndarray:
        # Specialized path for PE-L/14 and PE-G/14: encode entire windows as video clips
        if self.model_name == "pe-l14" or self.model_name == "pe-g14":
            # Use configured per-GPU batch size for video clips if provided
            pe_bs = self.pe_video_batch_size or min(batch_size, 8)
            outs = []
            for s in range(0, len(windows), pe_bs):
                batch_windows = windows[s : s + pe_bs]
                feats = self._encode_windows_pe(batch_windows)
                outs.append(feats)
            if len(outs) == 0:
                return np.zeros((0, self.embed_dim), dtype=np.float32)
            return torch.cat(outs, dim=0).numpy()

        # Image encoders: sample frames and average per window
        all_samples = []
        map_win_to_slice = []
        cursor = 0
        for w in windows:
            idxs = self._sample_indices(len(w), frames_per_window, random)
            if not idxs:
                all_samples.append(w[0])
                map_win_to_slice.append((cursor, cursor + 1))
                cursor += 1
                continue
            for j in idxs:
                all_samples.append(w[j])
            map_win_to_slice.append((cursor, cursor + len(idxs)))
            cursor += len(idxs)

        if self.model_name == "res50":
            encode_fn = self._encode_images_openai_clip
        elif self.model_name in {"clip", "siglip", "siglipl14", "siglip2"}:
            encode_fn = self._encode_images_hf
        elif self.model_name == "clip4clip":
            encode_fn = self._encode_images_clip4clip
        else:
            raise ValueError(f"Unknown model_name {self.model_name}")

        outs = []
        for s in range(0, len(all_samples), batch_size):
            feats = encode_fn(all_samples[s : s + batch_size])
            outs.append(feats)

        if len(outs) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        flat_feats = torch.cat(outs, dim=0)
        window_embs = []
        for a, b in map_win_to_slice:
            if b <= a:
                window_embs.append(torch.zeros(self.embed_dim))
            else:
                window_embs.append(flat_feats[a:b].mean(dim=0))
        return torch.stack(window_embs, dim=0).numpy()

    def _encode_windows_pe(self, windows: List[List[np.ndarray]]) -> torch.Tensor:
        """Encode a batch of windows (lists of BGR frames) as video clips.

        Pads each window in the batch to the batch's max temporal length by
        repeating the last frame so windows can be batched.
        Returns a CPU tensor of shape (B, D).
        """
        if not windows:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        # Preprocess each window: convert to tensors (T_i, C, H, W)
        clip_tensors = []
        max_T = 0
        for w in windows:
            if not w:
                # create a single black frame if window is empty
                black = np.zeros((336, 336, 3), dtype=np.uint8)
                w = [black]
            # If requested, uniformly sample to target temporal length
            if self.pe_target_T is not None and len(w) > 0:
                T = self.pe_target_T
                if len(w) >= T:
                    # uniform indices across [0, len(w)-1]
                    idxs = [int(round(i * (len(w) - 1) / (T - 1))) for i in range(T)]
                else:
                    # upsample by repeating last frame to reach T
                    idxs = list(range(len(w))) + [len(w) - 1] * (T - len(w))
                w = [w[i] for i in idxs]

            pil_imgs = [self._bgr_to_pil(f) for f in w]
            frames = [self.tokenizer(img) for img in pil_imgs]
            clip = torch.stack(frames, dim=0)
            clip_tensors.append(clip)
            max_T = max(max_T, clip.shape[0])

        # Pad all to max_T using last-frame repetition
        padded = []
        for clip in clip_tensors:
            if clip.shape[0] < max_T:
                pad = clip[-1:].expand(max_T - clip.shape[0], -1, -1, -1)
                clip = torch.cat([clip, pad], dim=0)
            padded.append(clip)

        x = torch.stack(padded, dim=0).to(self.device, non_blocking=True)  # (B,T,C,H,W)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device.type == "cuda"),
        ):
            feats = self.model.encode_video(x)
        return feats.float().detach().cpu()

    # ------------------------- labels
    def extract_labels(self, path: str) -> Optional[str]:
        if self.dataset_name == "breakfast":
            label = path.split("/")[-1]
            return label.split("_")[1].replace(".mp4", "")
        elif self.dataset_name == "ucf101":
            label = path.split("/")[-1]
            return label.split("_")[1]
        elif self.dataset_name == "hmdb":
            return path.split("/")[4]
        elif self.dataset_name == "something2":
            return path.split("/")[1]
        elif self.dataset_name == "jester":
            label = path.split("/")[-1]
            return label.split("_")[0]
        return None

    # ------------------------- main
    def embed_video(
        self,
        video_paths,
        window_size,
        output_path,
        random=True,
        save_intermediate=False,
        frames_per_window=1,
        batch_size=256,
    ):
        os.makedirs(output_path, exist_ok=True)

        video_embedding_paths, labels, video_window_spans, video_meta = {}, [], {}, {}

        video_paths = sorted(video_paths)
        save_base = os.path.join(
            output_path, f"{self.dataset_name}_{self.model_name}_{window_size}_state"
        )
        final_path = save_base + ".npy"
        tmp_path = save_base + ".tmp.npy"

        processed_count = 0
        if save_intermediate:
            load_path = (
                final_path
                if os.path.exists(final_path)
                else (tmp_path if os.path.exists(tmp_path) else None)
            )
            if load_path:
                try:
                    loaded = np.load(load_path, allow_pickle=True).item()
                    video_embedding_paths = loaded.get("video_embeddings", {})
                    labels = loaded.get("labels", [])
                    video_window_spans = loaded.get("video_window_spans", {})
                    video_meta = loaded.get("video_meta", {})
                    processed_count = len(video_embedding_paths)
                except Exception:
                    processed_count = 0
        if processed_count > 0:
            video_paths = video_paths[processed_count:]

        counter_since_last_save = 0
        for video_path in tqdm.tqdm(video_paths):
            labels.append(self.extract_labels(video_path))
            windows, spans, fps, read_frames = self._read_windows(
                video_path, window_size
            )

            if len(windows) == 0:
                video_embedding_paths[video_path] = np.zeros(
                    (0, self.embed_dim), dtype=np.float32
                )
                video_window_spans[video_path] = []
                video_meta[video_path] = {"fps": fps, "frame_count": float(read_frames)}
            else:
                window_embeddings = self._encode_windows(
                    windows, frames_per_window, random, batch_size
                )
                video_embedding_paths[video_path] = window_embeddings
                video_window_spans[video_path] = spans
                video_meta[video_path] = {"fps": fps, "frame_count": float(read_frames)}

            counter_since_last_save += 1
            if save_intermediate and (counter_since_last_save % 10 == 0):
                state = {
                    "video_embeddings": video_embedding_paths,
                    "labels": labels,
                    "video_window_spans": video_window_spans,
                    "video_meta": video_meta,
                }
                np.save(tmp_path, state, allow_pickle=True)
                os.replace(tmp_path, final_path)

        if save_intermediate:
            # delete tmp file if it exists
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(final_path):
                os.remove(final_path)

        self.video_embeddings = video_embedding_paths
        self.labels = labels
        self.video_window_spans = video_window_spans
        self.video_meta = video_meta

    def process_data(
        self,
        folder_path,
        window_size,
        output_path,
        random=True,
        save_intermediate=False,
        frames_per_window=1,
        batch_size=256,
    ):
        os.makedirs(output_path, exist_ok=True)
        video_paths = []
        if self.dataset_name == "jester":
            folder_path = folder_path[0].replace("Video_data", "Image_data")
            all_paths = os.listdir(folder_path)
            video_paths = [
                os.path.join(folder_path, p)
                for p in all_paths
                if os.path.isdir(os.path.join(folder_path, p))
            ]
        else:
            if isinstance(folder_path, list):
                for path in folder_path:
                    for root, _, files in os.walk(path):
                        for file in files:
                            if file.lower().endswith(".mp4"):
                                video_paths.append(os.path.join(root, file))
            else:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(".mp4"):
                            video_paths.append(os.path.join(root, file))
        print(len(video_paths), "videos found in", folder_path)

        self.embed_video(
            video_paths,
            window_size,
            output_path,
            random=random,
            save_intermediate=save_intermediate,
            frames_per_window=frames_per_window,
            batch_size=batch_size,
        )


class Create_Concepts(_PickleBackendsMixin):
    def __init__(self, model_name, model, tokenizer, clip_model=None):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_name = None
        self.video_embeddings = None
        self.labels = None
        self.text_concepts = None
        self.text_embeddings = None

    def embedd_text(self, *text):
        concepts = []

        # Case 1: multiple positional args given
        if len(text) > 1:
            for t in text:
                if isinstance(t, str):
                    concepts.extend([c.strip() for c in t.split(",") if c.strip()])
                elif isinstance(t, list):
                    for item in t:
                        concepts.extend(
                            [c.strip() for c in item.split(",") if c.strip()]
                        )
        else:
            t = text[0]
            if isinstance(t, str):
                concepts.extend([c.strip() for c in t.split(",") if c.strip()])
            elif isinstance(t, list):
                for item in t:
                    concepts.extend([c.strip() for c in item.split(",") if c.strip()])

        # Deduplicate while preserving order
        seen = set()
        concepts = [c for c in concepts if not (c in seen or seen.add(c))]
        # Tokenize & embed

        if self.model_name == "clip":
            inputs = self.tokenizer(
                concepts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)
            outputs = self.model.get_text_features(**inputs)

        elif self.model_name == "pe-l14" or self.model_name == "pe-g14":
            inputs = self.tokenizer(
                concepts).to(self.device)
            with torch.no_grad():
                outputs = self.model.encode_text(inputs)
        elif self.model_name == "siglip" or self.model_name == "siglipl14":
            inputs = self.tokenizer(
                text=concepts, padding="max_length", return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
        elif self.model_name == "siglip2":
            inputs = self.tokenizer(
                text=concepts, padding=True, return_tensors="pt"
            ).to(self.model.device)
            # text_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
        elif self.model_name == "res50":
            inputs = clip.tokenize(concepts)  # returns CPU tensor by default
            inputs = inputs.to(self.device)  # move tokens to model device
            with torch.no_grad():
                outputs = self.model.encode_text(inputs).detach().cpu()
        elif self.model_name == "clip4clip":
            inputs = self.tokenizer(
                concepts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)
            outputs = (
                self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )[0]
                .detach()
                .cpu()
            )
        else:
            outputs = None

        self.text_embeddings = outputs
        self.text_concepts = concepts
