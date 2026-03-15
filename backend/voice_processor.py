"""
VocalD v3  —  Production-Ready Speaker Identification System
=============================================================
Embedding stack : WavLM-Large (50%) + ECAPA-TDNN (30%) + ResNet (20%)
Diarization     : PyAnnote 3.1  (dual-strategy)
Backend         : PLDA + WCCN + Adaptive S-Norm + Multi-Centroid
Ensemble        : cosine/PLDA (85%) + Mahalanobis (15%)
Scheduler       : APScheduler daily retraining at 03:00
Storage         : SQLite (auto-created on first run)

Install
-------
pip install speechbrain transformers torchaudio pyannote.audio \
            webrtcvad librosa soundfile apscheduler python-dotenv scipy

Usage
-----
    from vocald_final import process_audio_file, extract_voice_embedding, \
                              find_matching_speaker, daily_model_update

    speakers = process_audio_file("meeting.wav", "meeting.wav")
    emb      = extract_voice_embedding("clip.wav")
    match    = find_matching_speaker(emb)
    daily_model_update(force=True)   # manual retrain
"""

# ── stdlib ────────────────────────────────────────────────────────────────
import os
import time
import pickle
import sqlite3
import logging
import threading
import tempfile
import multiprocessing
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import torch
import librosa
import soundfile as sf
import webrtcvad
from dotenv import load_dotenv
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# ── optional: PyAnnote ────────────────────────────────────────────────────
try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None

# ── optional: SpeechBrain ─────────────────────────────────────────────────
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        EncoderClassifier = None

# ── optional: WavLM (HuggingFace transformers) ────────────────────────────
try:
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    HAS_WAVLM = True
except ImportError:
    HAS_WAVLM = False

# ── optional: APScheduler ─────────────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False

load_dotenv()


# =============================================================================
# LOGGING
# =============================================================================

def _setup_logging() -> logging.Logger:
    fmt     = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler("vocald.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt,
                        datefmt=datefmt, handlers=handlers)
    return logging.getLogger("vocald")

log = _setup_logging()


# =============================================================================
# CONFIGURATION  — change values here, nowhere else
# =============================================================================

# Paths
DB_PATH               = os.getenv("VOCALD_DB",        "vocald.db")
PLDA_MODEL_PATH       = os.getenv("VOCALD_PLDA",       "vocald_plda.pkl")
WCCN_MODEL_PATH       = os.getenv("VOCALD_WCCN",       "vocald_wccn.pkl")
SCORE_NORM_MODEL_PATH = os.getenv("VOCALD_SNORM",      "vocald_score_norm.pkl")
THRESHOLD_STATE_PATH  = os.getenv("VOCALD_THRESH",     "vocald_thresholds.pkl")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Audio
SAMPLE_RATE           = 16_000
PLDA_EMBEDDING_DIM    = 256      # all models projected here

# Segment gates
MIN_SEGMENT_DURATION     = 0.3   # seconds — shorter segments are unreliable
MIN_SPEAKING_TIME        = 1.0   # minimum per-speaker duration after diarization
MAX_SEGMENTS_PER_SPEAKER = 15    # cap before outlier rejection
MIN_PROFILE_DURATION     = 1.5   # skip profiles shorter than this

# Similarity thresholds
BASE_SIMILARITY_THRESHOLD   = 0.45
BASE_STRONG_MATCH_THRESHOLD = 0.58
BASE_VERIFICATION_THRESHOLD = 0.50
ADAPTIVE_LEARNING_RATE      = (0.02, 0.10)

# Multi-centroid
ENABLE_MULTI_CENTROID         = True
MAX_CENTROIDS_PER_PROFILE     = 5
CENTROID_SIMILARITY_THRESHOLD = 0.70
CENTROID_MIN_SAMPLES          = 1
CENTROID_MERGE_THRESHOLD      = 0.82
CENTROID_QUALITY_WEIGHT       = 0.3

# Adaptive thresholds
ENABLE_ADAPTIVE_THRESHOLDS = False
THRESHOLD_MIN              = 0.38
THRESHOLD_MAX              = 0.75
THRESHOLD_STEP             = 0.02
PROFILE_MATURITY_THRESHOLD = 5
FALSE_ACCEPT_PENALTY       = 0.04
FALSE_REJECT_REWARD        = 0.02
SIMILARITY_GAP_THRESHOLD   = 0.15

# Embedding fusion (must sum to 1.0; auto-renormalised if a model is missing)
WAVLM_WEIGHT  = 0.50
ECAPA_WEIGHT  = 0.30
RESNET_WEIGHT = 0.20

# PLDA
ENABLE_PLDA           = True
PLDA_MIN_SPEAKERS     = 3
PLDA_LATENT_DIM       = 64
PLDA_RETRAIN_INTERVAL = 5
PLDA_SCORE_WEIGHT     = 0.65    # blend: PLDA * w + cosine * (1-w)

# WCCN
ENABLE_WCCN                  = True
WCCN_MIN_SPEAKERS            = 5
WCCN_MIN_SAMPLES_PER_SPEAKER = 2
WCCN_RETRAIN_INTERVAL        = 10
WCCN_REGULARIZATION          = 1e-4

# Score normalisation
ENABLE_SCORE_NORMALIZATION  = True
SCORE_NORM_METHOD           = "adaptive_snorm"   # zt_norm | adaptive_snorm | cohort
COHORT_SIZE                 = 100
ZT_NORM_SIZE                = 50
SNORM_TOP_N                 = 20
SCORE_NORM_MIN_SPEAKERS     = 8
SCORE_NORM_RETRAIN_INTERVAL = 15

# Ensemble scoring
ENABLE_MAHALANOBIS     = True
ENSEMBLE_COSINE_WEIGHT = 0.85
ENSEMBLE_MAHAL_WEIGHT  = 0.15

# Audio processing
ENABLE_VAD                   = True
VAD_AGGRESSIVENESS           = 3       # 0-3; 3 = most aggressive
ENABLE_GENDER_CLASSIFICATION = True
ENABLE_QUALITY_SCORING       = True
OUTLIER_REJECTION_FACTOR     = 3.5
ENABLE_NOISE_FILTERING       = True
LOW_FREQ_CUTOFF              = 85      # Hz high-pass
HIGH_FREQ_CUTOFF             = 7_600   # Hz low-pass

# Daily retraining
ENABLE_DAILY_RETRAINING = True
DAILY_RETRAIN_HOUR      = 3
DAILY_RETRAIN_MINUTE    = 0


# =============================================================================
# DATABASE  — schema auto-created on first run
# =============================================================================

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_profiles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL,
    embedding        BLOB    NOT NULL,
    first_seen       TEXT    NOT NULL,
    last_seen        TEXT    NOT NULL,
    total_recordings INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_voice_profiles_name
    ON voice_profiles (name);
"""

_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """Open a connection and guarantee the schema exists."""
    conn = sqlite3.connect(DB_PATH, timeout=15.0,
                           check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_DB_SCHEMA)
    conn.commit()
    return conn


def _collect_all_embeddings_from_db() -> dict:
    """
    Return {profile_id: [np.ndarray, ...]} of all stored fused embeddings.
    Handles both multi-centroid dicts and legacy flat profiles gracefully.
    """
    result: dict = {}
    try:
        with _db_lock:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT id, embedding FROM voice_profiles"
            ).fetchall()
            conn.close()

        for pid, blob in rows:
            data       = pickle.loads(blob)
            embeddings = []

            if isinstance(data, dict) and "centroids" in data:
                for c in data["centroids"]:
                    emb = c.get("fused_embedding") or c.get("resemblyzer_embedding")
                    if emb is not None:
                        embeddings.append(np.asarray(emb, dtype=np.float64))
            elif isinstance(data, dict):
                emb = data.get("fused_embedding")
                if emb is None:
                    emb = data.get("resemblyzer_embedding")
                if emb is None:
                    emb = data.get("embedding")
                if emb is not None:
                    embeddings.append(np.asarray(emb, dtype=np.float64))
            else:
                embeddings.append(np.asarray(data, dtype=np.float64))

            if embeddings:
                result[pid] = embeddings

    except Exception as exc:
        log.warning("_collect_all_embeddings_from_db: %s", exc)
    return result


# =============================================================================
# SCORE NORMALISATION
# =============================================================================

class ScoreNormalizer:
    """
    Implements three normalisation strategies:
        zt_norm         — Z-norm followed by T-norm
        adaptive_snorm  — Top-N cohort S-norm  (default, best in practice)
        cohort          — Simple impostor-mean/std shift
    """

    def __init__(self, method: str = "adaptive_snorm"):
        self.method                 = method
        self.is_trained             = False
        self.impostor_mean: float   = 0.5
        self.impostor_std:  float   = 0.15
        self.target_mean:   float   = 0.80
        self.target_std:    float   = 0.10
        self.cohort_embeddings: list = []
        self.speaker_stats:     dict = {}
        self._n_speakers_trained     = 0

    # ── training ─────────────────────────────────────────────────────────

    def train(self, speaker_embeddings: dict) -> None:
        n = len(speaker_embeddings)
        if n < SCORE_NORM_MIN_SPEAKERS:
            log.info("ScoreNorm.train: need >=%d speakers, have %d — skip",
                     SCORE_NORM_MIN_SPEAKERS, n)
            return

        all_e = [e for embs in speaker_embeddings.values() for e in embs]
        if len(all_e) > COHORT_SIZE:
            idx = np.random.choice(len(all_e), COHORT_SIZE, replace=False)
            self.cohort_embeddings = [all_e[i] for i in idx]
        else:
            self.cohort_embeddings = list(all_e)

        sids = list(speaker_embeddings.keys())
        rng  = np.random.default_rng()
        imp, tgt = [], []

        for _ in range(min(500, n * 10)):
            if n < 2:
                break
            s1, s2 = rng.choice(sids, 2, replace=False)
            e1 = speaker_embeddings[s1][rng.integers(0, len(speaker_embeddings[s1]))]
            e2 = speaker_embeddings[s2][rng.integers(0, len(speaker_embeddings[s2]))]
            imp.append(self._cos(e1, e2))

        for sid in sids:
            embs = speaker_embeddings[sid]
            if len(embs) >= 2:
                for _ in range(min(5, len(embs))):
                    i1, i2 = rng.choice(len(embs), 2, replace=False)
                    tgt.append(self._cos(embs[i1], embs[i2]))

        if imp:
            self.impostor_mean = float(np.mean(imp))
            self.impostor_std  = float(np.std(imp) + 1e-6)
        if tgt:
            self.target_mean = float(np.mean(tgt))
            self.target_std  = float(np.std(tgt) + 1e-6)

        self.speaker_stats = {}
        for sid, embs in speaker_embeddings.items():
            if len(embs) < 2:
                continue
            cs = [self._cos(e, ce)
                  for e  in embs[:min(5, len(embs))]
                  for ce in self.cohort_embeddings[:ZT_NORM_SIZE]]
            if cs:
                self.speaker_stats[sid] = {
                    "mean": float(np.mean(cs)),
                    "std":  float(np.std(cs) + 1e-6),
                }

        self.is_trained          = True
        self._n_speakers_trained = n
        log.info("ScoreNorm trained | method=%s | speakers=%d | "
                 "imp=%.3f+-%.3f  tgt=%.3f+-%.3f",
                 self.method, n,
                 self.impostor_mean, self.impostor_std,
                 self.target_mean,   self.target_std)

    # ── inference ─────────────────────────────────────────────────────────

    def normalize(self, raw: float,
                  test_emb=None, enroll_emb=None, speaker_id=None) -> float:
        if not self.is_trained:
            return raw
        if self.method == "zt_norm":
            return self._zt_norm(raw, test_emb, enroll_emb)
        if self.method == "adaptive_snorm":
            return self._adaptive_snorm(raw, test_emb)
        return self._cohort_norm(raw)

    def _zt_norm(self, raw, test_emb, enroll_emb) -> float:
        zs  = ([self._cos(test_emb, ce)
                for ce in self.cohort_embeddings[:ZT_NORM_SIZE]]
               if test_emb is not None and self.cohort_embeddings else [])
        zm  = float(np.mean(zs))  if zs else 0.0
        zsd = float(np.std(zs) + 1e-6) if zs else 1.0
        zn  = (raw - zm) / zsd

        if enroll_emb is not None and self.cohort_embeddings:
            ts  = [self._cos(enroll_emb, ce)
                   for ce in self.cohort_embeddings[:ZT_NORM_SIZE]]
            tm  = float(np.mean(ts))
            tsd = float(np.std(ts) + 1e-6)
            zn  = zn - (tm - zm) / tsd

        return float(np.clip(_sigmoid(zn), 0.0, 1.0))

    def _adaptive_snorm(self, raw, test_emb) -> float:
        if test_emb is None or not self.cohort_embeddings:
            return self._cohort_norm(raw)
        scores = sorted([self._cos(test_emb, ce) for ce in self.cohort_embeddings],
                        reverse=True)[:SNORM_TOP_N]
        if not scores:
            return raw
        sm, ss = float(np.mean(scores)), float(np.std(scores) + 1e-6)
        return float(np.clip(_sigmoid((raw - sm) / ss * 2.0), 0.0, 1.0))

    def _cohort_norm(self, raw) -> float:
        n = (raw - self.impostor_mean) / self.impostor_std
        return float(np.clip(_sigmoid(n * 2.0), 0.0, 1.0))

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _cos(a, b) -> float:
        a = np.asarray(a, np.float64); a = a / (np.linalg.norm(a) + 1e-8)
        b = np.asarray(b, np.float64); b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.clip(a @ b, -1.0, 1.0))

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str = SCORE_NORM_MODEL_PATH) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=4)

    def load(self, path: str = SCORE_NORM_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                self.__dict__.update(pickle.load(fh))
            log.info("ScoreNorm loaded | method=%s | speakers_at_train=%d",
                     self.method, self._n_speakers_trained)
            return True
        except Exception as exc:
            log.warning("ScoreNorm load error: %s", exc)
            return False


class ScoreNormManager:
    def __init__(self):
        self.normalizer = ScoreNormalizer(method=SCORE_NORM_METHOD)
        self._since     = 0
        if ENABLE_SCORE_NORMALIZATION and not self.normalizer.load():
            self._retrain()

    def _retrain(self):
        embs = _collect_all_embeddings_from_db()
        if len(embs) < SCORE_NORM_MIN_SPEAKERS:
            self.normalizer.is_trained = False
            return
        self.normalizer.train(embs)
        self.normalizer.save()
        self._since = 0

    def on_new_profile(self):
        self._since += 1
        if self._since >= SCORE_NORM_RETRAIN_INTERVAL:
            self._retrain()

    @property
    def is_ready(self) -> bool:
        return ENABLE_SCORE_NORMALIZATION and self.normalizer.is_trained

    def normalize(self, score: float, test_emb=None,
                  enroll_emb=None, speaker_id=None) -> float:
        return (self.normalizer.normalize(score, test_emb, enroll_emb, speaker_id)
                if self.is_ready else score)


# =============================================================================
# WCCN — Within-Class Covariance Normalisation
# =============================================================================

class WCCN:
    def __init__(self):
        self.W    = None
        self.mean = None
        self.is_trained              = False
        self._n_speakers_trained     = 0

    def train(self, speaker_embeddings: dict) -> None:
        valid = {sid: e for sid, e in speaker_embeddings.items()
                 if len(e) >= WCCN_MIN_SAMPLES_PER_SPEAKER}
        if len(valid) < WCCN_MIN_SPEAKERS:
            log.info("WCCN.train: need >=%d qualified speakers, have %d — skip",
                     WCCN_MIN_SPEAKERS, len(valid))
            return

        all_e = np.array([e for embs in valid.values() for e in embs],
                         dtype=np.float64)
        D         = all_e.shape[1]
        self.mean = all_e.mean(axis=0)

        Sw = np.zeros((D, D), dtype=np.float64)
        n  = 0
        for embs in valid.values():
            arr = np.asarray(embs, dtype=np.float64)
            c   = arr - arr.mean(axis=0)
            Sw += c.T @ c
            n  += len(embs)

        Sw = Sw / max(n, 1) + np.eye(D) * WCCN_REGULARIZATION
        ev, evec = np.linalg.eigh(Sw)
        ev       = np.maximum(ev, WCCN_REGULARIZATION)
        self.W   = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T

        self.is_trained          = True
        self._n_speakers_trained = len(valid)
        log.info("WCCN trained | speakers=%d | dim=%d", len(valid), D)

    def transform(self, emb: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.W is None:
            return emb
        out = self.W @ (emb.astype(np.float64) - self.mean)
        n   = np.linalg.norm(out)
        return out / n if n > 1e-8 else out

    def save(self, path: str = WCCN_MODEL_PATH) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"W": self.W, "mean": self.mean,
                         "is_trained": self.is_trained,
                         "n_speakers": self._n_speakers_trained}, fh, protocol=4)

    def load(self, path: str = WCCN_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                s = pickle.load(fh)
            self.W, self.mean, self.is_trained = s["W"], s["mean"], s["is_trained"]
            self._n_speakers_trained = s.get("n_speakers", 0)
            log.info("WCCN loaded | speakers_at_train=%d", self._n_speakers_trained)
            return True
        except Exception as exc:
            log.warning("WCCN load error: %s", exc)
            return False


class WCCNManager:
    def __init__(self):
        self.wccn   = WCCN()
        self._since = 0
        if ENABLE_WCCN and not self.wccn.load():
            self._retrain()

    def _retrain(self):
        embs = _collect_all_embeddings_from_db()
        if len(embs) < WCCN_MIN_SPEAKERS:
            self.wccn.is_trained = False
            return
        self.wccn.train(embs)
        self.wccn.save()
        self._since = 0

    def on_new_profile(self):
        self._since += 1
        if self._since >= WCCN_RETRAIN_INTERVAL:
            self._retrain()

    @property
    def is_ready(self) -> bool:
        return ENABLE_WCCN and self.wccn.is_trained

    def transform(self, emb: np.ndarray) -> np.ndarray:
        return self.wccn.transform(emb) if self.is_ready else emb


# =============================================================================
# MULTI-CENTROID VOICE PROFILE
# =============================================================================

class VoiceProfileCentroid:
    """
    Maintains up to MAX_CENTROIDS_PER_PROFILE acoustic centroids per speaker.
    New embeddings are routed to the nearest centroid (if close enough) or
    spawn a new one.  Centroids are periodically merged and pruned.
    """

    def __init__(self, max_centroids: int = MAX_CENTROIDS_PER_PROFILE):
        self.centroids:     list = []
        self.max_centroids: int  = max_centroids
        self.total_samples: int  = 0
        self.metadata: dict = {
            "created_at":   datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    # ── public API ────────────────────────────────────────────────────────

    def add_embedding(self, emb_data: dict,
                      quality: float = 1.0, duration: float = 1.0) -> None:
        self.total_samples += 1
        fused = _get_fused(emb_data)
        ecapa = emb_data.get("ecapa_embedding") if isinstance(emb_data, dict) else None

        if not self.centroids:
            self._new(fused, ecapa, quality, duration, emb_data)
        else:
            idx, sim = self._nearest(fused)
            if sim >= CENTROID_SIMILARITY_THRESHOLD:
                self._update(idx, fused, ecapa, quality, duration)
            elif len(self.centroids) < self.max_centroids:
                self._new(fused, ecapa, quality, duration, emb_data)
            else:
                self._update(idx, fused, ecapa, quality, duration)

        if self.total_samples % 10 == 0:
            self._merge()
            self._prune()

        self.metadata["last_updated"] = datetime.now().isoformat()

    def best_score(self, emb_data: dict,
                   plda_mgr=None, wccn_mgr=None,
                   norm_mgr=None, speaker_id=None) -> float:
        if not self.centroids:
            return 0.0

        fused = _get_fused(emb_data)
        if wccn_mgr and wccn_mgr.is_ready:
            fused = wccn_mgr.transform(fused)

        ecapa = emb_data.get("ecapa_embedding") if isinstance(emb_data, dict) else None

        scores, weights = [], []
        for c in self.centroids:
            cf = np.asarray(c["fused_embedding"], dtype=np.float64)
            if wccn_mgr and wccn_mgr.is_ready:
                cf = wccn_mgr.transform(cf)

            cos  = float(np.clip(fused @ cf, -1.0, 1.0))
            euc  = 1.0 / (1.0 + np.linalg.norm(fused - cf))
            pear = float(np.clip(np.corrcoef(fused, cf)[0, 1], -1.0, 1.0))
            fs   = 0.65 * cos + 0.25 * euc + 0.10 * pear

            ce = c.get("ecapa_embedding")
            if ecapa is not None and ce is not None:
                es   = float(np.clip(np.asarray(ecapa) @ np.asarray(ce), -1.0, 1.0))
                base = ECAPA_WEIGHT / (ECAPA_WEIGHT + RESNET_WEIGHT) * es + \
                       RESNET_WEIGHT / (ECAPA_WEIGHT + RESNET_WEIGHT) * fs
            else:
                base = fs

            if plda_mgr and plda_mgr.is_ready:
                pp   = plda_mgr.score_prob(fused, cf)
                sc   = PLDA_SCORE_WEIGHT * pp + (1.0 - PLDA_SCORE_WEIGHT) * base
            else:
                sc = base

            if norm_mgr and norm_mgr.is_ready:
                sc = norm_mgr.normalize(sc, test_emb=fused,
                                        enroll_emb=cf, speaker_id=speaker_id)

            scores.append(sc)
            weights.append(
                c["quality_avg"] ** CENTROID_QUALITY_WEIGHT
                * np.log1p(c["sample_count"])
            )

        sc_arr = np.asarray(scores)
        wt_arr = np.asarray(weights)
        wt_arr = wt_arr / wt_arr.sum()
        return float(0.85 * sc_arr.max() + 0.15 * np.average(sc_arr, weights=wt_arr))

    def primary(self):
        return max(self.centroids, key=lambda c: c["sample_count"]) \
               if self.centroids else None

    def to_dict(self) -> dict:
        return {"centroids":     self.centroids,
                "total_samples": self.total_samples,
                "metadata":      self.metadata}

    @classmethod
    def from_dict(cls, data: dict,
                  max_centroids: int = MAX_CENTROIDS_PER_PROFILE) -> "VoiceProfileCentroid":
        obj = cls(max_centroids)
        obj.centroids     = data.get("centroids", [])
        obj.total_samples = data.get("total_samples", 0)
        obj.metadata      = data.get("metadata", {
            "created_at":   datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        })
        return obj

    # ── internals ─────────────────────────────────────────────────────────

    def _nearest(self, emb: np.ndarray):
        sims = [
            0.7 * float(np.clip(emb @ np.asarray(c["fused_embedding"]), -1.0, 1.0))
            + 0.3 / (1.0 + np.linalg.norm(emb - np.asarray(c["fused_embedding"])))
            for c in self.centroids
        ]
        idx = int(np.argmax(sims))
        return idx, sims[idx]

    def _new(self, fused, ecapa, quality, duration, data):
        self.centroids.append({
            "fused_embedding": fused.copy(),
            "ecapa_embedding": ecapa.copy() if ecapa is not None else None,
            "sample_count":   1,
            "quality_avg":    float(quality),
            "weight":         float(quality * duration),
            "duration_total": float(duration),
            "gender":         data.get("gender", "unknown")
                              if isinstance(data, dict) else "unknown",
            "created_at":     datetime.now().isoformat(),
        })

    def _update(self, idx, fused, ecapa, quality, duration):
        c     = self.centroids[idx]
        alpha = (min(0.15, max(0.02, 1.0 / (c["sample_count"] + 1)))
                 * (0.7 + 0.3 * quality))

        nf = (1 - alpha) * np.asarray(c["fused_embedding"]) + alpha * fused
        nf /= (np.linalg.norm(nf) + 1e-8)
        c["fused_embedding"] = nf

        if ecapa is not None:
            ce = c.get("ecapa_embedding")
            if ce is not None:
                ne = (1 - alpha) * np.asarray(ce) + alpha * ecapa
                ne /= (np.linalg.norm(ne) + 1e-8)
                c["ecapa_embedding"] = ne
            else:
                c["ecapa_embedding"] = ecapa.copy()

        c["sample_count"]   += 1
        c["weight"]          += quality * duration
        c["quality_avg"]      = ((c["quality_avg"] * (c["sample_count"] - 1) + quality)
                                 / c["sample_count"])
        c["duration_total"]  += duration

    def _merge(self):
        i = 0
        while i < len(self.centroids) - 1:
            j = i + 1
            while j < len(self.centroids):
                fi = np.asarray(self.centroids[i]["fused_embedding"])
                fj = np.asarray(self.centroids[j]["fused_embedding"])
                if float(np.clip(fi @ fj, -1.0, 1.0)) >= CENTROID_MERGE_THRESHOLD:
                    ci, cj = self.centroids[i], self.centroids[j]
                    tot    = ci["sample_count"] + cj["sample_count"]
                    wi     = ci["sample_count"] / tot
                    wj     = cj["sample_count"] / tot
                    m      = wi * fi + wj * fj
                    m     /= (np.linalg.norm(m) + 1e-8)
                    ci["fused_embedding"] = m
                    if ci.get("ecapa_embedding") is not None and cj.get("ecapa_embedding") is not None:
                        me  = wi * np.asarray(ci["ecapa_embedding"]) + wj * np.asarray(cj["ecapa_embedding"])
                        me /= (np.linalg.norm(me) + 1e-8)
                        ci["ecapa_embedding"] = me
                    ci["sample_count"]   = tot
                    ci["weight"]        += cj["weight"]
                    ci["quality_avg"]    = wi * ci["quality_avg"] + wj * cj["quality_avg"]
                    ci["duration_total"] += cj["duration_total"]
                    del self.centroids[j]
                else:
                    j += 1
            i += 1

    def _prune(self):
        if len(self.centroids) <= 2:
            return
        med = float(np.median([c["sample_count"] for c in self.centroids]))
        self.centroids = [c for c in self.centroids
                          if c["sample_count"] >= max(CENTROID_MIN_SAMPLES, med * 0.15)]


# =============================================================================
# ADAPTIVE THRESHOLD MANAGER
# =============================================================================

class AdaptiveThresholdManager:
    def __init__(self):
        self.similarity_threshold   = BASE_SIMILARITY_THRESHOLD
        self.strong_match_threshold = BASE_STRONG_MATCH_THRESHOLD
        self.verification_threshold = BASE_VERIFICATION_THRESHOLD
        self.profile_thresholds:    dict = {}
        self._adjustments:          list = []
        self._load()

    def _load(self):
        if not ENABLE_ADAPTIVE_THRESHOLDS or not os.path.exists(THRESHOLD_STATE_PATH):
            return
        try:
            with open(THRESHOLD_STATE_PATH, "rb") as fh:
                s = pickle.load(fh)
            self.similarity_threshold   = s.get("similarity_threshold",   BASE_SIMILARITY_THRESHOLD)
            self.strong_match_threshold = s.get("strong_match_threshold", BASE_STRONG_MATCH_THRESHOLD)
            self.verification_threshold = s.get("verification_threshold", BASE_VERIFICATION_THRESHOLD)
            self.profile_thresholds     = s.get("profile_thresholds", {})
            log.info("Thresholds loaded | base=%.3f", self.similarity_threshold)
        except Exception as exc:
            log.warning("Threshold load error: %s", exc)

    def _save(self):
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return
        try:
            with open(THRESHOLD_STATE_PATH, "wb") as fh:
                pickle.dump({
                    "similarity_threshold":   self.similarity_threshold,
                    "strong_match_threshold": self.strong_match_threshold,
                    "verification_threshold": self.verification_threshold,
                    "profile_thresholds":     self.profile_thresholds,
                    "updated":                datetime.now().isoformat(),
                }, fh, protocol=4)
        except Exception as exc:
            log.warning("Threshold save error: %s", exc)

    def get_threshold(self, profile_id: int,
                      total_recordings: int, num_centroids: int = 1) -> float:
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return self.similarity_threshold
        base = self.profile_thresholds.get(profile_id, self.similarity_threshold)
        if total_recordings >= PROFILE_MATURITY_THRESHOLD:
            base += min((total_recordings - PROFILE_MATURITY_THRESHOLD) / 20.0, 0.10)
        if num_centroids > 2:
            base += min((num_centroids - 2) * 0.015, 0.05)
        return float(np.clip(base, THRESHOLD_MIN, THRESHOLD_MAX))

    def analyze(self, all_sims: list, best_id) -> None:
        if not ENABLE_ADAPTIVE_THRESHOLDS or len(all_sims) < 2:
            return
        top2 = sorted([s[0] for s in all_sims], reverse=True)[:2]
        gap  = top2[0] - top2[1]
        if gap > SIMILARITY_GAP_THRESHOLD and top2[0] > self.similarity_threshold + 0.05:
            self._shift(THRESHOLD_STEP, f"high-conf gap={gap:.3f}")
        elif gap < 0.05 and top2[0] < self.similarity_threshold + 0.03:
            self._shift(-THRESHOLD_STEP, f"low-conf  gap={gap:.3f}")
        n = self._db_count()
        if n > 20:
            target = BASE_SIMILARITY_THRESHOLD + min((n - 20) / 100.0, 0.08)
            if self.similarity_threshold < target - 0.01:
                self._shift(THRESHOLD_STEP, f"DB-scale n={n}")

    def on_match_confirmed(self, profile_id: int,
                           similarity: float, was_correct: bool) -> None:
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return
        cur = self.profile_thresholds.get(profile_id, self.similarity_threshold)
        if was_correct and similarity < cur + 0.05:
            new = cur - FALSE_REJECT_REWARD
        elif not was_correct and similarity > cur:
            new = cur + FALSE_ACCEPT_PENALTY
        else:
            return
        self.profile_thresholds[profile_id] = float(np.clip(new, THRESHOLD_MIN, THRESHOLD_MAX))
        self._save()

    def _shift(self, delta: float, reason: str) -> None:
        old = self.similarity_threshold
        self.similarity_threshold = float(np.clip(old + delta, THRESHOLD_MIN, THRESHOLD_MAX))
        if abs(self.similarity_threshold - old) < 1e-6:
            return
        self._adjustments.append((datetime.now().isoformat(), reason, delta))
        if len(self._adjustments) > 100:
            self._adjustments = self._adjustments[-100:]
        r = self.similarity_threshold / old if old > 0 else 1.0
        self.strong_match_threshold = float(
            np.clip(self.strong_match_threshold * r, THRESHOLD_MIN + 0.05, THRESHOLD_MAX))
        self.verification_threshold = float(
            np.clip(self.verification_threshold * r, THRESHOLD_MIN + 0.02, THRESHOLD_MAX - 0.05))
        self._save()
        log.info("Threshold: %.3f -> %.3f  (%s)", old, self.similarity_threshold, reason)

    def _db_count(self) -> int:
        try:
            with _db_lock:
                conn = _get_conn()
                n    = conn.execute("SELECT COUNT(*) FROM voice_profiles").fetchone()[0]
                conn.close()
            return n
        except Exception:
            return 0


threshold_manager = AdaptiveThresholdManager()


# =============================================================================
# PLDA — Probabilistic Linear Discriminant Analysis
# =============================================================================

class PLDA:
    """
    Full EM-trained PLDA with cached Cholesky inverses.
    Scoring cost is O(D) after the one-time O(D^3) cache at train time.
    """

    def __init__(self, embedding_dim: int = PLDA_EMBEDDING_DIM,
                 latent_dim: int = PLDA_LATENT_DIM):
        self.embedding_dim           = embedding_dim
        self.latent_dim              = latent_dim
        self.mean                    = None
        self.V                       = None
        self.Sw                      = None
        self.is_trained              = False
        self._n_speakers_trained     = 0
        self._llr_scale              = 1.0
        self._Sw_inv                 = None
        self._Sigmas_inv             = None
        self._logdet_w               = 0.0
        self._logdet_s               = 0.0

    # ── training ─────────────────────────────────────────────────────────

    def train(self, speaker_embeddings: dict, n_iter: int = 10) -> None:
        all_e = np.array([e for embs in speaker_embeddings.values()
                          for e in embs], dtype=np.float64)
        if all_e.shape[0] < 2:
            return
        D = all_e.shape[1]
        self.mean = all_e.mean(axis=0)

        Sw = np.zeros((D, D), dtype=np.float64)
        nt, sp_means = 0, {}
        for sid, embs in speaker_embeddings.items():
            arr = np.asarray(embs, dtype=np.float64)
            m   = arr.mean(axis=0)
            sp_means[sid] = m
            Sw += (arr - m).T @ (arr - m)
            nt += len(embs)
        Sw = Sw / max(nt, 1) + np.eye(D) * 1e-4

        sp_mat   = np.asarray(list(sp_means.values()), dtype=np.float64) - self.mean
        Sb       = (sp_mat.T @ sp_mat) / max(len(sp_means), 1)
        ev, evec = np.linalg.eigh(Sb)
        k        = min(self.latent_dim, D)
        V        = evec[:, np.argsort(ev)[::-1][:k]].T.copy()
        Sw_inv   = np.linalg.inv(Sw)

        for _ in range(n_iter):
            VSwi  = V @ Sw_inv
            Ainv  = np.linalg.inv(np.eye(k) + VSwi @ V.T)
            Ez, EzzT = {}, {}
            for sid, embs in speaker_embeddings.items():
                arr = np.asarray(embs, dtype=np.float64)
                ms  = arr.mean(axis=0) - self.mean
                ez  = Ainv @ (VSwi @ ms)
                Ez[sid]   = ez
                EzzT[sid] = Ainv + np.outer(ez, ez) * len(embs)

            nV   = np.zeros((k, D), dtype=np.float64)
            dV   = np.zeros((k, k), dtype=np.float64)
            Sw2  = np.zeros((D, D), dtype=np.float64)
            nt   = 0
            for sid, embs in speaker_embeddings.items():
                arr = np.asarray(embs, dtype=np.float64)
                ni  = len(embs)
                ms  = arr.mean(axis=0) - self.mean
                ez  = Ez[sid]
                nV  += ni * np.outer(ez, ms)
                dV  += EzzT[sid]
                Sw2 += (arr - (self.mean + V.T @ ez)).T \
                       @ (arr - (self.mean + V.T @ ez))
                nt  += ni

            V      = np.linalg.inv(dV + np.eye(k) * 1e-6) @ nV
            Sw     = Sw2 / max(nt, 1) + np.eye(D) * 1e-4
            Sw_inv = np.linalg.inv(Sw)

        self.V  = V
        self.Sw = Sw
        self.is_trained          = True
        self._n_speakers_trained = len(speaker_embeddings)
        self._llr_scale          = self._calibrate(speaker_embeddings)
        self._cache_inv()
        log.info("PLDA trained | speakers=%d | latent=%d | iters=%d | scale=%.3f",
                 len(speaker_embeddings), k, n_iter, self._llr_scale)

    def _cache_inv(self) -> None:
        if not self.is_trained or self.V is None or self.Sw is None:
            return
        D = self.embedding_dim
        try:
            self._Sw_inv = np.linalg.solve(self.Sw, np.eye(D))
            Ss           = self.Sw + self.V.T @ self.V
            L            = np.linalg.cholesky(Ss + np.eye(D) * 1e-6)
            self._Sigmas_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(D)))
            _, self._logdet_w = np.linalg.slogdet(self.Sw)
            _, self._logdet_s = np.linalg.slogdet(Ss)
        except np.linalg.LinAlgError as exc:
            log.warning("PLDA._cache_inv failed: %s — falling back to per-call inv", exc)
            self._Sw_inv = self._Sigmas_inv = None

    def _calibrate(self, speaker_embeddings: dict, n: int = 30) -> float:
        sids = list(speaker_embeddings.keys())
        if len(sids) < 2:
            return 1.0
        rng  = np.random.default_rng(0)
        llrs = []
        for _ in range(n):
            s = rng.choice(sids); d = s
            while d == s:
                d = rng.choice(sids)
            es = speaker_embeddings[s]
            i, j = rng.choice(len(es), 2, replace=(len(es) < 2))
            llrs.append(abs(self._raw(es[i], es[j])))
            llrs.append(abs(self._raw(es[0], speaker_embeddings[d][0])))
        return max(float(np.median(llrs)), 1e-6)

    def score(self, e1: np.ndarray, e2: np.ndarray) -> float:
        return self._raw(e1, e2) / self._llr_scale

    def score_prob(self, e1: np.ndarray, e2: np.ndarray) -> float:
        return _sigmoid(self.score(e1, e2))

    def _raw(self, e1: np.ndarray, e2: np.ndarray) -> float:
        if not self.is_trained or self.V is None or self.Sw is None:
            return 0.0
        x1 = e1.astype(np.float64) - self.mean
        x2 = e2.astype(np.float64) - self.mean

        if self._Sw_inv is not None:
            Si, Ssi = self._Sw_inv, self._Sigmas_inv
            ldw, lds = self._logdet_w, self._logdet_s
        else:
            D   = self.embedding_dim
            Si  = np.linalg.inv(self.Sw)
            Ss  = self.Sw + self.V.T @ self.V
            Ssi = np.linalg.inv(Ss + np.eye(D) * 1e-6)
            _, ldw = np.linalg.slogdet(self.Sw)
            _, lds = np.linalg.slogdet(Ss)

        qd = (x1 @ Si @ x1 + x2 @ Si @ x2) * 0.5
        qs = ((x1 - x2) @ Si @ (x1 - x2) + (x1 + x2) @ Ssi @ (x1 + x2)) * 0.25
        return float((ldw - lds) + (qd - qs))

    def save(self, path: str = PLDA_MODEL_PATH) -> None:
        with open(path, "wb") as fh:
            pickle.dump({
                "embedding_dim": self.embedding_dim,
                "latent_dim":    self.latent_dim,
                "mean": self.mean, "V": self.V, "Sw": self.Sw,
                "is_trained":    self.is_trained,
                "n_speakers":    self._n_speakers_trained,
                "llr_scale":     self._llr_scale,
            }, fh, protocol=4)

    def load(self, path: str = PLDA_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                s = pickle.load(fh)
            self.embedding_dim          = s["embedding_dim"]
            self.latent_dim             = s["latent_dim"]
            self.mean                   = s["mean"]
            self.V                      = s["V"]
            self.Sw                     = s.get("Sw") or s.get("Sigma_w")
            self.is_trained             = s["is_trained"]
            self._n_speakers_trained    = s.get("n_speakers", 0)
            self._llr_scale             = s.get("llr_scale", 1.0)
            self._cache_inv()
            log.info("PLDA loaded | speakers_at_train=%d", self._n_speakers_trained)
            return True
        except Exception as exc:
            log.warning("PLDA load error: %s", exc)
            return False


class PLDAManager:
    def __init__(self):
        self.plda   = PLDA()
        self._since = 0
        if ENABLE_PLDA and not self.plda.load():
            self._retrain()

    def _retrain(self):
        embs = _collect_all_embeddings_from_db()
        if len(embs) < PLDA_MIN_SPEAKERS:
            self.plda.is_trained = False
            return
        self.plda.train(embs)
        self.plda.save()
        self._since = 0

    def on_new_profile(self):
        self._since += 1
        if self._since >= PLDA_RETRAIN_INTERVAL:
            self._retrain()

    @property
    def is_ready(self) -> bool:
        return ENABLE_PLDA and self.plda.is_trained

    def score_llr(self, e1, e2) -> float:
        return self.plda.score(e1, e2) if self.is_ready else 0.0

    def score_prob(self, e1, e2) -> float:
        return self.plda.score_prob(e1, e2) if self.is_ready else 0.5


# =============================================================================
# SYSTEM INITIALISATION — models loaded once at import time
# =============================================================================

log.info("VocalD v3 starting — WavLM + ECAPA + ResNet")

# ── WavLM-Large ───────────────────────────────────────────────────────────
wavlm_model     = None
wavlm_extractor = None
if HAS_WAVLM:
    try:
        log.info("Loading WavLM-Large (~1.2 GB, first run downloads) ...")
        _wlm_name       = "microsoft/wavlm-large"
        wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_wlm_name)
        wavlm_model     = WavLMModel.from_pretrained(_wlm_name)
        wavlm_model.eval()
        log.info("WavLM-Large loaded  (1024-dim)")
    except Exception as _exc:
        log.error("WavLM load failed: %s", _exc)
        wavlm_model = wavlm_extractor = None
else:
    log.warning("WavLM unavailable — pip install transformers torchaudio")

# ── ECAPA-TDNN ────────────────────────────────────────────────────────────
ecapa_model = None
if EncoderClassifier is not None:
    try:
        log.info("Loading ECAPA-TDNN ...")
        ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"})
        log.info("ECAPA-TDNN loaded  (192-dim)")
    except Exception as _exc:
        log.error("ECAPA load failed: %s", _exc)

# ── ResNet ────────────────────────────────────────────────────────────────
resnet_model = None
if EncoderClassifier is not None:
    try:
        log.info("Loading ResNet speaker embedder ...")
        resnet_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb",
            savedir="pretrained_models/spkrec-resnet-voxceleb",
            run_opts={"device": "cpu"})
        log.info("ResNet embedder loaded  (256-dim)")
    except Exception as _exc:
        log.error("ResNet load failed: %s", _exc)

if wavlm_model is None and ecapa_model is None and resnet_model is None:
    log.error(
        "All three embedding models failed to load. "
        "Check your internet connection and pip install transformers speechbrain."
    )

# ── PyAnnote diarization ──────────────────────────────────────────────────
# FIX: allowlist TorchVersion for torch 2.5+ weights_only=True default
diarization_pipeline = None
if PyannotePipeline is not None:
    if not HF_TOKEN:
        log.warning("HUGGINGFACE_TOKEN not set — diarization unavailable. "
                    "Add it to .env to enable multi-speaker support.")
    else:
        try:
            import torch.serialization
            try:
                from torch.torch_version import TorchVersion
                torch.serialization.add_safe_globals([TorchVersion])
                log.info("Registered TorchVersion as safe global for torch.load")
            except Exception as _tv_exc:
                log.warning("Could not register TorchVersion safe global: %s", _tv_exc)

            diarization_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN)
            diarization_pipeline.to(torch.device("cpu"))
            torch.set_num_threads(multiprocessing.cpu_count())
            torch.set_grad_enabled(False)
            log.info("PyAnnote 3.1 loaded  (%d CPU cores)", multiprocessing.cpu_count())
        except Exception as _exc:
            log.error("PyAnnote load failed: %s", _exc)

# ── VAD ───────────────────────────────────────────────────────────────────
_vad = None
if ENABLE_VAD:
    try:
        _vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        log.info("VAD enabled  (aggressiveness=%d)", VAD_AGGRESSIVENESS)
    except Exception:
        log.warning("webrtcvad not available — VAD disabled")

# ── Backend managers ──────────────────────────────────────────────────────
log.info("Initialising PLDA ...")
plda_manager = PLDAManager()

log.info("Initialising WCCN ...")
wccn_manager = WCCNManager()

log.info("Initialising ScoreNorm ...")
score_norm_manager = ScoreNormManager()

log.info("=" * 60)
log.info("  WavLM-Large   : %s", "OK  (1024-dim)" if wavlm_model  else "MISSING")
log.info("  ECAPA-TDNN    : %s", "OK  (192-dim)"  if ecapa_model  else "MISSING")
log.info("  ResNet        : %s", "OK  (256-dim)"  if resnet_model else "MISSING")
log.info("  Diarization   : %s", "OK" if diarization_pipeline else "MISSING (set HF token)")
log.info("  PLDA          : %s", "trained" if plda_manager.is_ready  else "waiting for data")
log.info("  WCCN          : %s", "trained" if wccn_manager.is_ready  else "waiting for data")
log.info("  ScoreNorm     : %s", "trained" if score_norm_manager.is_ready else "waiting for data")
log.info("  Mahalanobis   : %s", "on" if ENABLE_MAHALANOBIS else "off")
log.info("  Daily retrain : %s", "on  (%02d:%02d)" % (DAILY_RETRAIN_HOUR, DAILY_RETRAIN_MINUTE)
         if ENABLE_DAILY_RETRAINING and HAS_APSCHEDULER else "off")
log.info("=" * 60)


# =============================================================================
# AUDIO UTILITY FUNCTIONS
# =============================================================================

def _apply_vad(wav: np.ndarray) -> np.ndarray:
    """Strip non-speech frames using WebRTC VAD."""
    if not ENABLE_VAD or _vad is None:
        return wav
    try:
        frame_ms  = 30
        frame_len = int(SAMPLE_RATE * frame_ms / 1000)
        i16       = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        kept      = [
            wav[i: i + frame_len]
            for i in range(0, len(i16) - frame_len, frame_len)
            if _vad.is_speech(i16[i: i + frame_len].tobytes(), SAMPLE_RATE)
        ]
        return np.concatenate(kept) if kept else wav
    except Exception as exc:
        log.debug("VAD error: %s", exc)
        return wav


def _noise_filter(wav: np.ndarray) -> np.ndarray:
    """DC removal, bandpass filter, Gaussian smoothing, normalise."""
    try:
        wav = wav - wav.mean()
        mx  = np.abs(wav).max()
        if mx > 0:
            wav = wav / mx
        if not ENABLE_NOISE_FILTERING:
            return wav
        ny = SAMPLE_RATE / 2.0
        lo = LOW_FREQ_CUTOFF / ny
        if 0.0 < lo < 1.0:
            b, a = butter(5, lo, btype="high")
            wav  = filtfilt(b, a, wav)
        hi = HIGH_FREQ_CUTOFF / ny
        if 0.0 < hi < 1.0:
            b, a = butter(5, hi, btype="low")
            wav  = filtfilt(b, a, wav)
        wav = gaussian_filter1d(wav, sigma=0.8)
        mx  = np.abs(wav).max()
        return wav / mx if mx > 0 else wav
    except Exception as exc:
        log.debug("Noise filter error: %s", exc)
        wav = wav - wav.mean()
        mx  = np.abs(wav).max()
        return wav / mx if mx > 0 else wav


def _audio_quality(wav: np.ndarray) -> float:
    """Estimate recording quality in [0, 1]."""
    if not ENABLE_QUALITY_SCORING or len(wav) < SAMPLE_RATE:
        return 1.0
    try:
        nn         = min(int(0.2 * SAMPLE_RATE), len(wav) // 4)
        noise_pwr  = float(np.mean(wav[:nn] ** 2)) + 1e-10
        speech_pwr = float(np.mean(wav[nn:] ** 2)) + 1e-10

        if noise_pwr > speech_pwr * 0.5:
            snr_score = float(np.clip(np.std(wav) / 0.10, 0.0, 1.0))
        else:
            snr       = 10.0 * np.log10(speech_pwr / noise_pwr)
            snr_score = float(np.clip((snr - 5.0) / 20.0, 0.0, 1.0))

        dr_score   = float(np.clip(np.std(wav) / 0.15, 0.0, 1.0))
        clip_score = 1.0 - float(np.clip(
            np.sum(np.abs(wav) > 0.95) / len(wav) * 50.0, 0.0, 1.0))
        zcr        = float(np.sum(np.abs(np.diff(np.sign(wav)))) / len(wav))
        zcr_score  = float(np.clip(zcr / 0.1, 0.0, 1.0))

        return 0.40 * snr_score + 0.25 * dr_score + 0.25 * clip_score + 0.10 * zcr_score
    except Exception:
        return 0.5


def _classify_gender(wav: np.ndarray) -> str:
    """Estimate speaker gender via F0 median."""
    if not ENABLE_GENDER_CLASSIFICATION or len(wav) < SAMPLE_RATE:
        return "unknown"
    try:
        f0 = librosa.yin(wav, fmin=75, fmax=400, sr=SAMPLE_RATE)
        v  = f0[f0 > 75]
        if len(v) < 10:
            return "unknown"
        m = float(np.median(v))
        return "male" if m < 145 else ("female" if m > 165 else "unknown")
    except Exception:
        return "unknown"


def _project_to_dim(emb: np.ndarray, target_dim: int = PLDA_EMBEDDING_DIM) -> np.ndarray:
    """Project any embedding to target_dim deterministically."""
    e = np.asarray(emb, dtype=np.float64).ravel()
    d = target_dim
    if len(e) == d:
        return e
    if len(e) > d:
        while len(e) > d:
            half   = len(e) // 2
            rest   = len(e) - half
            minlen = min(half, rest)
            e      = (e[:minlen] + e[len(e) - minlen:]) / np.sqrt(2.0)
        e = e[:d]
    else:
        out       = np.zeros(d, dtype=np.float64)
        out[:len(e)] = e
        e         = out
    n = np.linalg.norm(e)
    return e / n if n > 1e-8 else e


def _encode_speechbrain(model, wav: np.ndarray):
    """Run a SpeechBrain EncoderClassifier on a 16 kHz waveform."""
    try:
        t = torch.FloatTensor(wav).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(t).squeeze().cpu().numpy()
        n = float(np.linalg.norm(emb))
        return (emb / (n + 1e-8)).astype(np.float64) if n > 1e-8 else None
    except Exception as exc:
        log.debug("SpeechBrain encode error: %s", exc)
        return None


def _encode_wavlm(wav: np.ndarray):
    """Extract mean-pooled WavLM-Large speaker embedding (1024-dim)."""
    if wavlm_model is None or wavlm_extractor is None:
        return None
    try:
        inputs = wavlm_extractor(
            wav, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True)
        with torch.no_grad():
            out = wavlm_model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        n   = float(np.linalg.norm(emb))
        return (emb / (n + 1e-8)).astype(np.float64) if n > 1e-8 else None
    except Exception as exc:
        log.debug("WavLM encode error: %s", exc)
        return None


# =============================================================================
# HELPERS
# =============================================================================

def _get_fused(data) -> np.ndarray:
    """Extract the primary fused embedding vector from any stored format."""
    if isinstance(data, dict):
        emb = data.get("fused_embedding")
        if emb is None:
            emb = data.get("resemblyzer_embedding")
        if emb is None:
            emb = data.get("embedding")
    else:
        emb = data
    return np.asarray(emb, dtype=np.float64).ravel()


def _to_centroid(stored) -> VoiceProfileCentroid:
    """Always return a VoiceProfileCentroid regardless of stored format."""
    if isinstance(stored, dict) and "centroids" in stored:
        return VoiceProfileCentroid.from_dict(stored)
    obj = VoiceProfileCentroid(max_centroids=1)
    obj.add_embedding(stored, quality=1.0, duration=1.0)
    return obj


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


# =============================================================================
# MAHALANOBIS DISTANCE CACHE
# =============================================================================

_mahal_inv_cov = None
_mahal_cache_n: int = 0
_mahal_lock         = threading.Lock()


def _get_inv_cov():
    """Return regularised inverse covariance; rebuilt only when DB grows."""
    global _mahal_inv_cov, _mahal_cache_n
    if not ENABLE_MAHALANOBIS:
        return None
    try:
        with _db_lock:
            conn  = _get_conn()
            count = conn.execute("SELECT COUNT(*) FROM voice_profiles").fetchone()[0]
            conn.close()
    except Exception:
        return _mahal_inv_cov
    if count < 5:
        return None
    if count == _mahal_cache_n and _mahal_inv_cov is not None:
        return _mahal_inv_cov
    with _mahal_lock:
        if count == _mahal_cache_n and _mahal_inv_cov is not None:
            return _mahal_inv_cov
        vecs = [e for embs in _collect_all_embeddings_from_db().values()
                for e in embs]
        if len(vecs) < 10:
            return None
        X = np.asarray(vecs, dtype=np.float64)
        try:
            _mahal_inv_cov = np.linalg.inv(
                np.cov(X.T) + np.eye(X.shape[1]) * 1e-4)
            _mahal_cache_n = count
        except np.linalg.LinAlgError:
            _mahal_inv_cov = None
    return _mahal_inv_cov


def _mahal_sim(a: np.ndarray, b: np.ndarray, ic: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(1.0 / (1.0 + np.sqrt(max(float(d @ ic @ d), 0.0))))


# =============================================================================
# CORE FUNCTION 1 — extract_voice_embedding()
# =============================================================================

def extract_voice_embedding(audio_path: str,
                             start_time=None,
                             end_time=None):
    """
    Extract a fused 256-dim speaker embedding from an audio file or segment.
    Returns dict or None if audio is too short or all models fail.
    """
    try:
        wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                              dtype=np.float32)
        wav    = wav.astype(np.float64)

        if start_time is not None and end_time is not None:
            s = max(0, int(start_time * SAMPLE_RATE))
            e = min(len(wav), int(end_time * SAMPLE_RATE))
            if s >= e:
                return None
            wav = wav[s:e]

        if len(wav) < int(MIN_SEGMENT_DURATION * SAMPLE_RATE):
            return None

        wav = _apply_vad(wav.astype(np.float32)).astype(np.float64)
        if len(wav) < int(MIN_SEGMENT_DURATION * SAMPLE_RATE):
            return None

        wav     = _noise_filter(wav)
        quality = _audio_quality(wav)
        gender  = _classify_gender(wav)

        wav_f32 = wav.astype(np.float32)

        wavlm_emb  = _encode_wavlm(wav_f32)
        ecapa_emb  = _encode_speechbrain(ecapa_model,  wav_f32) \
                     if ecapa_model  is not None else None
        resnet_emb = _encode_speechbrain(resnet_model, wav_f32) \
                     if resnet_model is not None else None

        if wavlm_emb is None and ecapa_emb is None and resnet_emb is None:
            log.warning("All models returned None for %s", audio_path)
            return None

        w_wlm = WAVLM_WEIGHT  if wavlm_emb  is not None else 0.0
        w_eca = ECAPA_WEIGHT  if ecapa_emb  is not None else 0.0
        w_rsn = RESNET_WEIGHT if resnet_emb is not None else 0.0
        total = w_wlm + w_eca + w_rsn

        fused = np.zeros(PLDA_EMBEDDING_DIM, dtype=np.float64)
        if wavlm_emb  is not None:
            fused += (w_wlm / total) * _project_to_dim(wavlm_emb)
        if ecapa_emb  is not None:
            fused += (w_eca / total) * _project_to_dim(ecapa_emb)
        if resnet_emb is not None:
            fused += (w_rsn / total) * _project_to_dim(resnet_emb)

        n = np.linalg.norm(fused)
        fused /= (n + 1e-8)

        return {
            "fused_embedding":  fused,
            "ecapa_embedding":  ecapa_emb,
            "resnet_embedding": resnet_emb,
            "wavlm_embedding":  wavlm_emb,
            "quality":          float(quality),
            "gender":           gender,
            "duration":         float(len(wav) / SAMPLE_RATE),
        }

    except Exception as exc:
        log.error("extract_voice_embedding(%s): %s", audio_path, exc)
        return None


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_speaker_match(embedding_data: dict, profile_id: int):
    """Secondary verification gate — three independent checks."""
    try:
        with _db_lock:
            conn = _get_conn()
            row  = conn.execute(
                "SELECT embedding, total_recordings FROM voice_profiles WHERE id=?",
                (profile_id,)
            ).fetchone()
            conn.close()

        if not row:
            return False, 0.0

        stored   = pickle.loads(row[0])
        total    = int(row[1])
        cobj     = _to_centroid(stored)
        num_cent = len(cobj.centroids)

        sim = cobj.best_score(
            embedding_data,
            plda_mgr=plda_manager,
            wccn_mgr=wccn_manager,
            norm_mgr=score_norm_manager,
            speaker_id=profile_id,
        )
        thr = threshold_manager.get_threshold(profile_id, total, num_cent)

        new_f = _get_fused(embedding_data)
        prim  = cobj.primary()
        old_f = _get_fused(prim) if prim is not None else new_f

        if wccn_manager.is_ready:
            new_f = wccn_manager.transform(new_f)
            old_f = wccn_manager.transform(old_f)

        cos_ok  = float(np.clip(new_f @ old_f, -1.0, 1.0)) >= 0.40
        euc_ok  = (1.0 / (1.0 + np.linalg.norm(new_f - old_f))) >= 0.35
        plda_ok = (plda_manager.score_llr(new_f, old_f) >= 0.0) \
                  if plda_manager.is_ready else True

        passed = sim >= thr and cos_ok and euc_ok and plda_ok
        return passed, float(sim)

    except Exception as exc:
        log.error("verify_speaker_match(id=%d): %s", profile_id, exc)
        return False, 0.0


# =============================================================================
# CORE FUNCTION 2 — find_matching_speaker()
# =============================================================================

def find_matching_speaker(embedding_data: dict):
    """Identify the best matching speaker profile."""
    try:
        with _db_lock:
            conn     = _get_conn()
            profiles = conn.execute(
                "SELECT id, name, embedding, total_recordings FROM voice_profiles"
            ).fetchall()
            conn.close()
    except Exception as exc:
        log.error("find_matching_speaker DB read: %s", exc)
        return None

    if not profiles:
        return None

    inv_cov = _get_inv_cov()
    fused_t = _get_fused(embedding_data)
    if wccn_manager.is_ready:
        fused_t = wccn_manager.transform(fused_t)

    all_sims  = []
    best_match = None
    best_sim   = 0.0

    for pid, name, blob, total_rec in profiles:
        try:
            stored = pickle.loads(blob)
        except Exception:
            continue

        cobj     = _to_centroid(stored)
        num_cent = len(cobj.centroids)

        cosine_score = cobj.best_score(
            embedding_data,
            plda_mgr=plda_manager,
            wccn_mgr=wccn_manager,
            norm_mgr=score_norm_manager,
            speaker_id=pid,
        )

        mahal_score = 0.0
        if inv_cov is not None:
            prim = cobj.primary()
            if prim is not None:
                cf = np.asarray(prim["fused_embedding"], dtype=np.float64)
                if wccn_manager.is_ready:
                    cf = wccn_manager.transform(cf)
                mahal_score = _mahal_sim(fused_t, cf, inv_cov)

        if inv_cov is not None:
            denom    = ENSEMBLE_COSINE_WEIGHT + ENSEMBLE_MAHAL_WEIGHT
            ensemble = (ENSEMBLE_COSINE_WEIGHT * cosine_score
                        + ENSEMBLE_MAHAL_WEIGHT * mahal_score) / denom
        else:
            ensemble = cosine_score

        all_sims.append((float(ensemble), pid, name))
        threshold = threshold_manager.get_threshold(pid, total_rec, num_cent)

        if ensemble > threshold and ensemble > best_sim:
            verified, vsim = verify_speaker_match(embedding_data, pid)
            if verified:
                best_sim   = float(ensemble)
                best_match = {
                    "id":            pid,
                    "name":          name,
                    "confidence":    round(float(ensemble) * 100, 2),
                    "similarity":    float(ensemble),
                    "verified":      True,
                    "num_centroids": num_cent,
                }

    threshold_manager.analyze(all_sims, best_match["id"] if best_match else None)
    return best_match


# =============================================================================
# CORE FUNCTION 3 — daily_model_update()
# =============================================================================

def daily_model_update(force: bool = False) -> dict:
    """Retrain PLDA, WCCN, and ScoreNorm from all current DB data."""
    t0  = time.time()
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 60)
    log.info("DAILY MODEL UPDATE  —  %s", ts)
    log.info("=" * 60)

    result = {
        "plda_retrained":       False,
        "wccn_retrained":       False,
        "score_norm_retrained": False,
        "num_speakers":         0,
        "duration_seconds":     0.0,
        "timestamp":            datetime.now().isoformat(),
    }

    embs = _collect_all_embeddings_from_db()
    n    = len(embs)
    result["num_speakers"] = n
    log.info("Speakers in DB: %d", n)

    if n >= PLDA_MIN_SPEAKERS or force:
        try:
            log.info("Retraining PLDA ...")
            plda_manager.plda.train(embs)
            plda_manager.plda.save()
            plda_manager.plda._cache_inv()
            plda_manager._since = 0
            result["plda_retrained"] = True
            log.info("PLDA retrained OK")
        except Exception as exc:
            log.error("PLDA retrain failed: %s", exc)
    else:
        log.info("PLDA skipped (need >=%d, have %d)", PLDA_MIN_SPEAKERS, n)

    qualified = {sid: ev for sid, ev in embs.items()
                 if len(ev) >= WCCN_MIN_SAMPLES_PER_SPEAKER}
    if len(qualified) >= WCCN_MIN_SPEAKERS or force:
        try:
            log.info("Retraining WCCN ...")
            wccn_manager.wccn.train(embs)
            wccn_manager.wccn.save()
            wccn_manager._since = 0
            global _mahal_cache_n
            _mahal_cache_n = 0
            result["wccn_retrained"] = True
            log.info("WCCN retrained OK")
        except Exception as exc:
            log.error("WCCN retrain failed: %s", exc)
    else:
        log.info("WCCN skipped (need >=%d qualified speakers)", WCCN_MIN_SPEAKERS)

    if n >= SCORE_NORM_MIN_SPEAKERS or force:
        try:
            log.info("Retraining ScoreNorm ...")
            score_norm_manager.normalizer.train(embs)
            score_norm_manager.normalizer.save()
            score_norm_manager._since = 0
            result["score_norm_retrained"] = True
            log.info("ScoreNorm retrained OK")
        except Exception as exc:
            log.error("ScoreNorm retrain failed: %s", exc)
    else:
        log.info("ScoreNorm skipped (need >=%d, have %d)",
                 SCORE_NORM_MIN_SPEAKERS, n)

    result["duration_seconds"] = round(time.time() - t0, 2)
    log.info(
        "Update complete in %.1fs | PLDA=%s WCCN=%s SN=%s",
        result["duration_seconds"],
        result["plda_retrained"],
        result["wccn_retrained"],
        result["score_norm_retrained"],
    )
    log.info("=" * 60)
    return result


# ── Schedule daily retraining ──────────────────────────────────────────────
_scheduler = None
if ENABLE_DAILY_RETRAINING and HAS_APSCHEDULER:
    try:
        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.add_job(
            daily_model_update,
            trigger="cron",
            hour=DAILY_RETRAIN_HOUR,
            minute=DAILY_RETRAIN_MINUTE,
            id="daily_model_update",
            replace_existing=True,
        )
        _scheduler.start()
        log.info("Daily retraining scheduled at %02d:%02d",
                 DAILY_RETRAIN_HOUR, DAILY_RETRAIN_MINUTE)
    except Exception as exc:
        log.warning("APScheduler start failed: %s", exc)


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def create_voice_profile(name: str, embedding_data: dict):
    """Create a new speaker profile. Returns the new profile id."""
    try:
        now = datetime.now().isoformat()
        if ENABLE_MULTI_CENTROID:
            obj = VoiceProfileCentroid()
            obj.add_embedding(
                embedding_data,
                quality=float(embedding_data.get("quality", 1.0)),
                duration=float(embedding_data.get("duration", 1.0)),
            )
            stored = obj.to_dict()
        else:
            stored = embedding_data

        with _db_lock:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO voice_profiles "
                "(name, embedding, first_seen, last_seen, total_recordings) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, pickle.dumps(stored, protocol=4), now, now, 1),
            )
            pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()
            conn.close()

        plda_manager.on_new_profile()
        wccn_manager.on_new_profile()
        score_norm_manager.on_new_profile()
        global _mahal_cache_n
        _mahal_cache_n = 0

        log.info("Created profile | id=%d  name='%s'", pid, name)
        return pid

    except Exception as exc:
        log.error("create_voice_profile: %s", exc)
        return None


def update_voice_profile(profile_id: int, new_data: dict) -> None:
    """Add new embedding data to an existing profile."""
    try:
        with _db_lock:
            conn = _get_conn()
            row  = conn.execute(
                "SELECT embedding, total_recordings FROM voice_profiles WHERE id=?",
                (profile_id,)
            ).fetchone()
            if not row:
                conn.close()
                return

            stored = pickle.loads(row[0])
            total  = int(row[1])

            if ENABLE_MULTI_CENTROID:
                obj = _to_centroid(stored)
                obj.add_embedding(
                    new_data,
                    quality=float(new_data.get("quality", 1.0)),
                    duration=float(new_data.get("duration", 1.0)),
                )
                refined = obj.to_dict()
            else:
                alpha  = (max(ADAPTIVE_LEARNING_RATE[0],
                              min(1.0 / (total + 1), ADAPTIVE_LEARNING_RATE[1]))
                          * float(new_data.get("quality", 1.0)))
                old_f  = _get_fused(stored)
                new_f  = _get_fused(new_data)
                merged = (1 - alpha) * old_f + alpha * new_f
                merged /= (np.linalg.norm(merged) + 1e-8)
                refined = {
                    "fused_embedding": merged,
                    "ecapa_embedding": new_data.get("ecapa_embedding"),
                    "quality":         float(new_data.get("quality", 1.0)),
                    "gender":          new_data.get("gender", "unknown"),
                }

            conn.execute(
                "UPDATE voice_profiles "
                "SET last_seen=?, total_recordings=total_recordings+1, embedding=? "
                "WHERE id=?",
                (datetime.now().isoformat(), pickle.dumps(refined, protocol=4), profile_id),
            )
            conn.commit()
            conn.close()

    except Exception as exc:
        log.error("update_voice_profile(id=%d): %s", profile_id, exc)


def rename_speaker(profile_id: int, new_name: str) -> bool:
    """Rename an existing speaker profile."""
    try:
        with _db_lock:
            conn = _get_conn()
            conn.execute(
                "UPDATE voice_profiles SET name=? WHERE id=?",
                (new_name, profile_id))
            conn.commit()
            conn.close()
        log.info("Renamed profile id=%d to '%s'", profile_id, new_name)
        return True
    except Exception as exc:
        log.error("rename_speaker: %s", exc)
        return False


def delete_speaker(profile_id: int) -> bool:
    """Delete a speaker profile from the database."""
    try:
        with _db_lock:
            conn = _get_conn()
            conn.execute("DELETE FROM voice_profiles WHERE id=?", (profile_id,))
            conn.commit()
            conn.close()
        global _mahal_cache_n
        _mahal_cache_n = 0
        log.info("Deleted profile id=%d", profile_id)
        return True
    except Exception as exc:
        log.error("delete_speaker: %s", exc)
        return False


def list_speakers() -> list:
    """Return all speaker profiles as a list of dicts."""
    try:
        with _db_lock:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT id, name, first_seen, last_seen, total_recordings "
                "FROM voice_profiles ORDER BY name"
            ).fetchall()
            conn.close()
        return [
            {"id": r[0], "name": r[1], "first_seen": r[2],
             "last_seen": r[3], "total_recordings": r[4]}
            for r in rows
        ]
    except Exception as exc:
        log.error("list_speakers: %s", exc)
        return []


# =============================================================================
# SEGMENT PROCESSING
# =============================================================================

def _merge_segments(segments: list, gap_s: float = 0.15) -> list:
    """Merge adjacent diarization segments separated by less than gap_s seconds."""
    if not segments:
        return segments
    segs   = sorted(segments, key=lambda x: x["start"])
    merged = [dict(segs[0])]
    for s in segs[1:]:
        if s["start"] - merged[-1]["end"] <= gap_s:
            merged[-1]["end"]      = s["end"]
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(dict(s))
    return merged


def _process_speaker_segments(args: tuple):
    """Worker — extract and average embeddings for one diarized speaker."""
    label, info, audio_path = args
    total_dur = float(info["total_duration"])

    segs = _merge_segments(info["segments"])
    segs = sorted(segs, key=lambda x: x["duration"], reverse=True)
    segs = segs[:MAX_SEGMENTS_PER_SPEAKER]

    embs:    list = []
    weights: list = []

    for seg in segs:
        if seg["duration"] < MIN_SEGMENT_DURATION:
            continue
        e = extract_voice_embedding(audio_path, seg["start"], seg["end"])
        if e is not None:
            embs.append(e)
            weights.append(seg["duration"] * float(e.get("quality", 1.0)))

    if not embs:
        return None

    if len(embs) > 3:
        fvecs = np.array([e["fused_embedding"] for e in embs], dtype=np.float64)
        med   = np.median(fvecs, axis=0)
        dists = [float(np.linalg.norm(v - med)) for v in fvecs]
        md    = float(np.median(dists))
        keep  = [(e, w) for e, w, d in zip(embs, weights, dists)
                 if d < md * OUTLIER_REJECTION_FACTOR]
        if keep:
            embs, weights = map(list, zip(*keep))

    wn   = np.asarray(weights, dtype=np.float64)
    wn   = wn / wn.sum()

    avgf = np.average(
        np.array([e["fused_embedding"] for e in embs], dtype=np.float64),
        axis=0, weights=wn)
    avgf /= (np.linalg.norm(avgf) + 1e-8)

    avg_ecapa = None
    ecapa_pairs = [(e["ecapa_embedding"], w)
                   for e, w in zip(embs, wn)
                   if e.get("ecapa_embedding") is not None]
    if ecapa_pairs:
        evecs, ews = zip(*ecapa_pairs)
        ews        = np.asarray(ews, dtype=np.float64)
        ews        = ews / ews.sum()
        avg_ecapa  = np.average(
            np.array(list(evecs), dtype=np.float64), axis=0, weights=ews)
        avg_ecapa /= (np.linalg.norm(avg_ecapa) + 1e-8)

    genders = [e.get("gender", "unknown") for e in embs]
    mode_g  = max(set(genders), key=genders.count)

    return {
        "label":   label,
        "embedding": {
            "fused_embedding":  avgf,
            "ecapa_embedding":  avg_ecapa,
            "quality":          float(np.mean([e.get("quality", 1.0) for e in embs])),
            "gender":           mode_g,
            "duration":         total_dur,
        },
        "duration":       total_dur,
        "num_embeddings": len(embs),
    }


# =============================================================================
# DIARIZATION
# =============================================================================

def _safe_tmpfile(audio_path: str) -> str:
    """Create a unique temp WAV path."""
    suffix = Path(audio_path).stem[:20]
    fd, path = tempfile.mkstemp(prefix=f"vocald_{suffix}_", suffix=".wav")
    os.close(fd)
    return path


def perform_smart_diarization(audio_path: str):
    """Run PyAnnote diarization with automatic two-strategy fallback."""
    if diarization_pipeline is None:
        return None

    tmp = None
    try:
        log.info("Diarizing: %s", Path(audio_path).name)
        y, sr    = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
        duration = float(len(y) / sr)
        log.info("Duration: %.1f s", duration)

        tmp = _safe_tmpfile(audio_path)
        sf.write(tmp, y, SAMPLE_RATE, subtype="PCM_16")

        with torch.no_grad():
            out = diarization_pipeline(tmp)

        spk: dict = {}
        for seg, _, lbl in out.itertracks(yield_label=True):
            spk.setdefault(lbl, []).append({
                "start":    seg.start,
                "end":      seg.end,
                "duration": seg.end - seg.start,
            })
        log.info("Strategy 1: %d speaker(s)", len(spk))

        needs_retry = (
            (len(spk) == 1 and duration > 15)
            or (duration > 180 and len(spk) < 3)
        )
        if needs_retry:
            log.info("Strategy 2: forcing min_speakers=2 ...")
            with torch.no_grad():
                out = diarization_pipeline(tmp, min_speakers=2, max_speakers=20)
            spk = {}
            for seg, _, lbl in out.itertracks(yield_label=True):
                spk.setdefault(lbl, []).append({
                    "start":    seg.start,
                    "end":      seg.end,
                    "duration": seg.end - seg.start,
                })
            log.info("Strategy 2: %d speaker(s)", len(spk))

        filtered = {
            lbl: {
                "segments":       segs,
                "total_duration": sum(s["duration"] for s in segs),
            }
            for lbl, segs in spk.items()
            if sum(s["duration"] for s in segs) >= MIN_SPEAKING_TIME
        }
        log.info("After duration filter: %d speaker(s)", len(filtered))
        return filtered or None

    except Exception as exc:
        log.error("perform_smart_diarization: %s", exc)
        import traceback
        traceback.print_exc()
        return None

    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# =============================================================================
# MAIN PIPELINE — process_audio_file()
# =============================================================================

def process_audio_file(audio_path: str, filename: str) -> list:
    """Full pipeline: diarize -> embed -> identify/enroll each speaker."""
    log.info("=" * 60)
    log.info("PROCESSING: %s", filename)
    log.info("=" * 60)
    t0       = time.time()
    speakers: list = []
    stem     = Path(filename).stem

    try:
        spk_data = perform_smart_diarization(audio_path)

        if not spk_data:
            log.info("Single-speaker mode (no diarization output)")
            emb = extract_voice_embedding(audio_path)

            if emb is None:
                return [{"speaker_index": 0, "name": f"Unknown ({stem})",
                         "confidence": 0.0, "voice_profile_id": None}]

            if emb["duration"] < MIN_PROFILE_DURATION:
                log.warning("Audio too short: %.1f s < %.1f s",
                            emb["duration"], MIN_PROFILE_DURATION)
                return [{"speaker_index": 0, "name": f"Too Short ({stem})",
                         "confidence": 0.0, "voice_profile_id": None}]

            match = find_matching_speaker(emb)
            if match:
                log.info("MATCHED: %s  (%.1f%%)", match["name"], match["confidence"])
                speakers.append({
                    "speaker_index":    0,
                    "name":             match["name"],
                    "confidence":       round(match["confidence"], 1),
                    "voice_profile_id": match["id"],
                })
                if (match["verified"]
                        and match["similarity"]
                        >= threshold_manager.strong_match_threshold):
                    update_voice_profile(match["id"], emb)
            else:
                name = f"Speaker {stem}"
                pid  = create_voice_profile(name, emb)
                log.info("NEW profile: '%s'  id=%s", name, pid)
                speakers.append({
                    "speaker_index":    0,
                    "name":             name,
                    "confidence":       95.0,
                    "voice_profile_id": pid,
                })

        else:
            log.info("Multi-speaker: %d speaker(s) detected", len(spk_data))

            args  = [
                (lbl, info, audio_path)
                for lbl, info in sorted(
                    spk_data.items(),
                    key=lambda x: x[1]["total_duration"],
                    reverse=True,
                )
            ]
            max_w = min(len(args), multiprocessing.cpu_count())

            with ThreadPoolExecutor(max_workers=max_w) as pool:
                future_map = {
                    pool.submit(_process_speaker_segments, a): a[0]
                    for a in args
                }
                raw_results = [f.result() for f in as_completed(future_map)]

            results = sorted(
                [r for r in raw_results if r is not None],
                key=lambda x: x["duration"],
                reverse=True,
            )

            idx = 0
            for result in results:
                emb = result["embedding"]
                if result["duration"] < MIN_PROFILE_DURATION:
                    log.info("  %s: %.1f s — too short, skipping",
                             result["label"], result["duration"])
                    continue

                log.info("  %s: %.1f s  (%d segs)  gender=%s  q=%.0f%%",
                         result["label"], result["duration"],
                         result["num_embeddings"],
                         emb.get("gender", "?"),
                         emb.get("quality", 0) * 100)

                match = find_matching_speaker(emb)
                if match:
                    log.info("  -> MATCHED: %s  (%.1f%%)",
                             match["name"], match["confidence"])
                    speakers.append({
                        "speaker_index":    idx,
                        "name":             match["name"],
                        "confidence":       round(match["confidence"], 1),
                        "voice_profile_id": match["id"],
                    })
                    if (match["verified"]
                            and match["similarity"]
                            >= threshold_manager.strong_match_threshold):
                        update_voice_profile(match["id"], emb)
                else:
                    name = f"Speaker {idx + 1}"
                    pid  = create_voice_profile(name, emb)
                    log.info("  -> NEW: '%s'  id=%s", name, pid)
                    speakers.append({
                        "speaker_index":    idx,
                        "name":             name,
                        "confidence":       95.0,
                        "voice_profile_id": pid,
                    })
                idx += 1

    except Exception as exc:
        log.error("process_audio_file error: %s", exc)
        import traceback
        traceback.print_exc()
        return [{"speaker_index": 0, "name": f"Error ({stem})",
                 "confidence": 0.0, "voice_profile_id": None}]

    log.info("COMPLETED: %d speaker(s) in %.1f s", len(speakers), time.time() - t0)
    return speakers


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    log.info("VocalD v3 ready.")
    log.info("Quick-start:")
    log.info("  daily_model_update(force=True)")
    log.info("  emb      = extract_voice_embedding('clip.wav')")
    log.info("  match    = find_matching_speaker(emb)")
    log.info("  speakers = process_audio_file('meeting.wav', 'meeting.wav')")
    log.info("  print(list_speakers())")