# camus_preprocessor.py
import os
import time
from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk


class CAMUSPreprocessor:
    """
    Preprocess CAMUS dataset:
      1) Build (or reuse) splits (train / test_ED / test_ES) based on Info_2CH.cfg quality.
      2) Load NIfTI frames (ED/ES) with SimpleITK.
      3) Ultrasound-friendly preprocessing: percentile clip -> CLAHE (opt) -> speckle denoise (opt).
      4) Resize to square (e.g., 256x256).
      5) Sanitize masks to labels {0,1,2,3}.
      6) Save per-sample .npz files (image float32 [0,1], mask uint8).

    Use:
        pp = CAMUSPreprocessor(data_dir="database_nifti", split_dir="prepared_data", out_dir="preprocessed")
        pp.build_splits_if_missing()
        pp.preprocess_all()
    """

    def __init__(
        self,
        data_dir: str = "database_nifti",
        split_dir: str = "prepared_data",
        out_dir: str = "preprocessed",
        view: str = "2CH",             # "2CH" or "4CH"
        img_size: int = 256,
        do_clahe: bool = True,
        denoise: str = "median",       # "median" | "bilateral" | None
        overwrite: bool = False,
        seed: int = 1234
    ):
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.out_dir = out_dir
        self.view = view
        self.img_size = int(img_size)
        self.do_clahe = bool(do_clahe)
        self.denoise = denoise
        self.overwrite = bool(overwrite)
        self.seed = int(seed)

        os.makedirs(self.split_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    # -------------------- public API --------------------

    def build_splits_if_missing(self):
        """Create train/test split .npy files if they don't exist."""
        train_p = self._p("train_samples.npy")
        test_ed = self._p("test_ED.npy")
        test_es = self._p("test_ES.npy")
        if all(Path(p).is_file() for p in [train_p, test_ed, test_es]):
            print("----- The split files already exist -----")
            return

        high, med, low = self._partition_by_quality(self.data_dir)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(high); rng.shuffle(med); rng.shuffle(low)

        # You can tune these counts
        h_s, m_s, l_s = 20, 20, 10
        htest, htrain = high[:h_s], high[h_s:]
        mtest, mtrain = med[:m_s],  med[m_s:]
        ltest, ltrain = low[:l_s],  low[l_s:]

        total_test  = np.array(htest + mtest + ltest, dtype=object)
        total_train = np.array(htrain + mtrain + ltrain, dtype=object)

        test_ED, test_ES = self._expand_rep(total_test, mode="separate")
        train_all        = self._expand_rep(total_train, mode="combined")

        np.save(train_p, np.array(train_all, dtype=object))
        np.save(test_ed, np.array(test_ED,  dtype=object))
        np.save(test_es, np.array(test_ES,  dtype=object))
        print("Splits created:",
              f"train={len(train_all)}, test_ED={len(test_ED)}, test_ES={len(test_ES)}")

    def preprocess_all(self):
        """Preprocess all splits into .npz files."""
        train_list = self._p("train_samples.npy")
        test_ed    = self._p("test_ED.npy")
        test_es    = self._p("test_ES.npy")

        self._preprocess_split(train_list, "train")
        self._preprocess_split(test_ed,    "test_ED")
        self._preprocess_split(test_es,    "test_ES")

        # Summary
        print("\nSummary:")
        for split in ["train", "test_ED", "test_ES"]:
            d = os.path.join(self.out_dir, split)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {split}: {n} files in {d}")

    # -------------------- internal helpers --------------------

    def _p(self, name: str) -> str:
        return os.path.join(self.split_dir, name)

    @staticmethod
    def _prepare_cfg(path: str):
        with open(path, "r") as f:
            lines = f.read().splitlines()
        out = {}
        for ln in lines:
            if ": " in ln:
                k, v = ln.split(": ", 1)
                out[k] = v
        return out

    def _partition_by_quality(self, data_dir):
        high, med, low = [], [], []
        for patient in sorted(os.listdir(data_dir)):
            pdir = os.path.join(data_dir, patient)
            if not os.path.isdir(pdir):
                continue
            info_path = os.path.join(pdir, "Info_2CH.cfg")  # quality key is in 2CH file
            if not os.path.isfile(info_path):
                print(f"warn: missing Info_2CH.cfg for {patient}, skipping")
                continue
            meta = self._prepare_cfg(info_path)
            q = meta.get("ImageQuality", None)
            if q == "Good":
                high.append(patient)
            elif q == "Medium":
                med.append(patient)
            elif q == "Poor":
                low.append(patient)
            else:
                print(f"warn: unknown ImageQuality '{q}' for {patient}")
        print(f"Quality counts -> High: {len(high)} | Medium: {len(med)} | Low: {len(low)}")
        return high, med, low

    @staticmethod
    def _expand_rep(arr, mode="combined"):
        """Expand patient IDs to [patient, 'ED'/'ES'] pairs."""
        if mode == "combined":
            out = []
            for a in arr:
                out += [[a, "ED"], [a, "ES"]]
            return out
        else:
            ed, es = [], []
            for a in arr:
                ed.append([a, "ED"])
                es.append([a, "ES"])
            return ed, es

    @staticmethod
    def _sitk_read(filepath: str):
        im = sitk.ReadImage(str(filepath))
        return np.squeeze(sitk.GetArrayFromImage(im))

    @staticmethod
    def _percentile_clip(img: np.ndarray, p_low=1.0, p_high=99.0):
        lo, hi = np.percentile(img, [p_low, p_high])
        if hi <= lo:
            return img.astype(np.float32)
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo + 1e-8)
        return img.astype(np.float32)

    @staticmethod
    def _apply_clahe_8bit(img01: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)):
        img8 = (img01 * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        out8 = clahe.apply(img8)
        return (out8.astype(np.float32) / 255.0)

    @staticmethod
    def _despeckle(img01: np.ndarray, method="median", ksize=3):
        if method == "median":
            out = cv2.medianBlur((img01 * 255).astype(np.uint8), ksize).astype(np.float32) / 255.0
        elif method == "bilateral":
            out = cv2.bilateralFilter(img01.astype(np.float32), d=5, sigmaColor=25, sigmaSpace=5)
        else:
            out = img01
        return out

    def _robust_preprocess_us(self, img: np.ndarray):
        """Percentile clip -> (CLAHE) -> (denoise). Output float32 in [0,1]."""
        if img.ndim == 3:
            img = np.squeeze(img)
        img = self._percentile_clip(img, 1.0, 99.0)
        if self.do_clahe:
            img = self._apply_clahe_8bit(img, clip_limit=2.0, tile_grid_size=(8, 8))
        img = self._despeckle(img, method=self.denoise, ksize=3)
        return img

    @staticmethod
    def _sanitize_mask(msk: np.ndarray):
        msk = msk.astype(np.int64)
        valid = np.array([0, 1, 2, 3], dtype=np.int64)
        okay = np.isin(msk, valid)
        if not okay.all():
            msk = np.where(okay, msk, 0)
        return msk

    def _resize_image_mask(self, img01: np.ndarray, msk: np.ndarray):
        size = self.img_size
        img_r = cv2.resize((img01 * 255.0).astype(np.uint8), (size, size),
                           interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        msk_r = cv2.resize(msk.astype(np.int32), (size, size),
                           interpolation=cv2.INTER_NEAREST).astype(np.int64)
        return img_r, msk_r

    def _load_pair(self, patient: str, instant: str):
        img_p = os.path.join(self.data_dir, patient, f"{patient}_{self.view}_{instant}.nii.gz")
        msk_p = os.path.join(self.data_dir, patient, f"{patient}_{self.view}_{instant}_gt.nii.gz")
        if not (os.path.isfile(img_p) and os.path.isfile(msk_p)):
            raise FileNotFoundError(f"Missing image or mask for {patient} {self.view} {instant}")
        img = self._sitk_read(img_p)
        msk = self._sitk_read(msk_p)
        return img, msk

    @staticmethod
    def _sid(patient: str, instant: str) -> str:
        return f"{patient}_{instant}"

    def _already_done(self, split_name, patient, instant):
        sid = self._sid(patient, instant)
        return os.path.isfile(os.path.join(self.out_dir, split_name, f"{sid}.npz"))

    def _save_npz(self, split_name, patient, instant, img01, msk):
        os.makedirs(os.path.join(self.out_dir, split_name), exist_ok=True)
        sid = self._sid(patient, instant)
        out_path = os.path.join(self.out_dir, split_name, f"{sid}.npz")
        np.savez_compressed(
            out_path,
            image=img01.astype(np.float32),  # [H,W] in [0,1]
            mask=msk.astype(np.uint8),       # [H,W] labels 0..3
            meta=np.array([patient, instant], dtype=object),
        )
        return out_path

    def _preprocess_split(self, split_path: str, split_name: str):
        items = np.load(split_path, allow_pickle=True)
        n = len(items)
        print(f"\n==> Preprocessing split '{split_name}' ({n} samples) ...")
        ok, fail = 0, 0
        t0 = time.time()

        for i, (patient, instant) in enumerate(items):
            try:
                if (not self.overwrite) and self._already_done(split_name, patient, instant):
                    ok += 1
                    if (i + 1) % 50 == 0:
                        print(f"[{split_name}] {i+1}/{n} (skipped existing) ...")
                    continue

                img, msk = self._load_pair(patient, instant)
                img = self._robust_preprocess_us(img)
                msk = self._sanitize_mask(msk)
                img, msk = self._resize_image_mask(img, msk)
                self._save_npz(split_name, patient, instant, img, msk)

                ok += 1
                if (i + 1) % 50 == 0:
                    print(f"[{split_name}] {i+1}/{n} processed ...")
            except Exception as e:
                fail += 1
                print(f"[ERROR] {patient} {instant}: {e}")

        dt = time.time() - t0
        print(f"Done '{split_name}': success={ok}, failed={fail}, time={dt:.1f}s")


# -------------------- CLI entrypoint --------------------

def main():
    pp = CAMUSPreprocessor(
        data_dir="database_nifti",
        split_dir="prepared_data",
        out_dir="preprocessed",
        view="2CH",            # change to "4CH" for 4-chamber
        img_size=256,
        do_clahe=True,
        denoise="median",
        overwrite=False,
        seed=1234,
    )
    pp.build_splits_if_missing()
    pp.preprocess_all()

    # Example: load one processed file to verify
    train_dir = os.path.join(pp.out_dir, "train")
    files = sorted(os.listdir(train_dir)) if os.path.isdir(train_dir) else []
    if files:
        sample = os.path.join(train_dir, files[0])
        arr = np.load(sample, allow_pickle=True)
        img = arr["image"]; msk = arr["mask"]; meta = arr["meta"]
        print(f"\nSample: {sample}")
        print("  image:", img.shape, img.dtype, float(img.min()), float(img.max()))
        print("  mask: ", msk.shape, msk.dtype, msk.min(), msk.max())
        print("  meta: ", meta)

if __name__ == "__main__":
    main()
