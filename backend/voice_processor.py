import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pyannote.audio import Pipeline
import sqlite3
from datetime import datetime
import pickle
from pathlib import Path
import torch
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import librosa
import soundfile as sf
import webrtcvad
import tempfile

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        EncoderClassifier = None

# =========================================================================
# CONFIGURATION
# =========================================================================

load_dotenv()

DB_PATH = 'vocald.db'
PLDA_MODEL_PATH = 'vocald_plda.pkl'
WCCN_MODEL_PATH = 'vocald_wccn.pkl'
SCORE_NORM_MODEL_PATH = 'vocald_score_norm.pkl'
THRESHOLD_STATE_PATH = 'vocald_thresholds.pkl'
SAMPLE_RATE = 16000

# SEGMENT / DURATION GATES
MIN_SEGMENT_DURATION = 0.4
MIN_SPEAKING_TIME = 1.5
MAX_SEGMENTS_PER_SPEAKER = 15
MIN_PROFILE_DURATION = 2.0

# BASE SIMILARITY THRESHOLDS
BASE_SIMILARITY_THRESHOLD = 0.75
BASE_STRONG_MATCH_THRESHOLD = 0.82
BASE_VERIFICATION_THRESHOLD = 0.78
ADAPTIVE_LEARNING_RATE = (0.02, 0.10)

# CENTROID CLUSTERING SETTINGS
ENABLE_MULTI_CENTROID = True
MAX_CENTROIDS_PER_PROFILE = 5
CENTROID_SIMILARITY_THRESHOLD = 0.82
CENTROID_MIN_SAMPLES = 2
CENTROID_MERGE_THRESHOLD = 0.90
CENTROID_QUALITY_WEIGHT = 0.3

# ADAPTIVE THRESHOLD SETTINGS
ENABLE_ADAPTIVE_THRESHOLDS = False
THRESHOLD_MIN = 0.65
THRESHOLD_MAX = 0.88
THRESHOLD_STEP = 0.02
PROFILE_MATURITY_THRESHOLD = 5
FALSE_ACCEPT_PENALTY = 0.04
FALSE_REJECT_REWARD = -0.02
SIMILARITY_GAP_THRESHOLD = 0.15

# ECAPA-TDNN
ENABLE_ECAPA = True
ECAPA_WEIGHT = 0.55

# PLDA SETTINGS
ENABLE_PLDA = True
PLDA_MIN_SPEAKERS = 3
PLDA_EMBEDDING_DIM = 256
PLDA_LATENT_DIM = 64
PLDA_RETRAIN_INTERVAL = 5
PLDA_SCORE_WEIGHT = 0.70

# WCCN SETTINGS
ENABLE_WCCN = True
WCCN_MIN_SPEAKERS = 5
WCCN_MIN_SAMPLES_PER_SPEAKER = 2
WCCN_RETRAIN_INTERVAL = 10
WCCN_REGULARIZATION = 1e-4

# SCORE NORMALIZATION SETTINGS
ENABLE_SCORE_NORMALIZATION = True
SCORE_NORM_METHOD = 'adaptive_snorm'
COHORT_SIZE = 100
ZT_NORM_SIZE = 50
SNORM_TOP_N = 20
SCORE_NORM_MIN_SPEAKERS = 8
SCORE_NORM_RETRAIN_INTERVAL = 15

# ADVANCED FEATURES
ENABLE_VAD = True
ENABLE_GENDER_CLASSIFICATION = True
ENABLE_QUALITY_SCORING = True
OUTLIER_REJECTION_FACTOR = 2.5

# NOISE FILTERING
ENABLE_NOISE_FILTERING = True
LOW_FREQ_CUTOFF = 85
HIGH_FREQ_CUTOFF = 8000


# =========================================================================
# SCORE NORMALIZATION SYSTEM
# =========================================================================

class ScoreNormalizer:
    def __init__(self, method='adaptive_snorm'):
        self.method = method
        self.is_trained = False
        self.impostor_mean = None
        self.impostor_std = None
        self.target_mean = None
        self.target_std = None
        self.cohort_embeddings = []
        self.speaker_stats = {}
        self._num_speakers_at_train = 0

    def train(self, speaker_embeddings):
        if len(speaker_embeddings) < SCORE_NORM_MIN_SPEAKERS:
            print(f"   ScoreNorm: need >= {SCORE_NORM_MIN_SPEAKERS} speakers. "
                  f"Current: {len(speaker_embeddings)}")
            return

        all_embeddings = [e for embs in speaker_embeddings.values() for e in embs]
        if len(all_embeddings) > COHORT_SIZE:
            indices = np.random.choice(len(all_embeddings), COHORT_SIZE, replace=False)
            self.cohort_embeddings = [all_embeddings[i] for i in indices]
        else:
            self.cohort_embeddings = all_embeddings.copy()

        impostor_scores = []
        target_scores = []
        speaker_ids = list(speaker_embeddings.keys())
        rng = np.random.default_rng()

        for _ in range(min(500, len(speaker_ids) * 10)):
            if len(speaker_ids) < 2:
                break
            sid1, sid2 = rng.choice(speaker_ids, 2, replace=False)
            embs1 = speaker_embeddings[sid1]
            embs2 = speaker_embeddings[sid2]
            impostor_scores.append(self._raw_similarity(
                embs1[rng.integers(0, len(embs1))],
                embs2[rng.integers(0, len(embs2))]))

        for sid in speaker_ids:
            embs = speaker_embeddings[sid]
            if len(embs) >= 2:
                for _ in range(min(5, len(embs))):
                    idx1, idx2 = rng.choice(len(embs), 2, replace=False)
                    target_scores.append(self._raw_similarity(embs[idx1], embs[idx2]))

        if impostor_scores and target_scores:
            self.impostor_mean = float(np.mean(impostor_scores))
            self.impostor_std = float(np.std(impostor_scores) + 1e-6)
            self.target_mean = float(np.mean(target_scores))
            self.target_std = float(np.std(target_scores) + 1e-6)
        else:
            self.impostor_mean = 0.5
            self.impostor_std = 0.15
            self.target_mean = 0.8
            self.target_std = 0.10

        for sid, embs in speaker_embeddings.items():
            if len(embs) < 2:
                continue
            cohort_scores = [self._raw_similarity(emb, ce)
                             for emb in embs[:min(5, len(embs))]
                             for ce in self.cohort_embeddings[:ZT_NORM_SIZE]]
            if cohort_scores:
                self.speaker_stats[sid] = {
                    'mean': float(np.mean(cohort_scores)),
                    'std': float(np.std(cohort_scores) + 1e-6)
                }

        self.is_trained = True
        self._num_speakers_at_train = len(speaker_embeddings)
        print(f"   ScoreNorm trained | method={self.method} | speakers={len(speaker_embeddings)}")

    def normalize(self, raw_score, test_emb=None, enroll_emb=None, speaker_id=None):
        if not self.is_trained:
            return raw_score
        if self.method == 'zt_norm':
            return self._zt_norm(raw_score, test_emb, enroll_emb)
        elif self.method == 'adaptive_snorm':
            return self._adaptive_snorm(raw_score, test_emb)
        return self._cohort_norm(raw_score)

    def _zt_norm(self, raw_score, test_emb, enroll_emb):
        z_scores = ([self._raw_similarity(test_emb, ce)
                     for ce in self.cohort_embeddings[:ZT_NORM_SIZE]]
                    if test_emb is not None and self.cohort_embeddings else [])
        if z_scores:
            z_mean = np.mean(z_scores)
            z_std = np.std(z_scores) + 1e-6
            z_norm = (raw_score - z_mean) / z_std
        else:
            z_norm = raw_score
        t_norm = z_norm
        if enroll_emb is not None and self.cohort_embeddings:
            t_scores = [self._raw_similarity(enroll_emb, ce)
                        for ce in self.cohort_embeddings[:ZT_NORM_SIZE]]
            if t_scores:
                t_norm = z_norm - (np.mean(t_scores) - (np.mean(z_scores) if z_scores else 0)) / (np.std(t_scores) + 1e-6)
        return float(np.clip(1.0 / (1.0 + np.exp(-t_norm)), 0.0, 1.0))

    def _adaptive_snorm(self, raw_score, test_emb):
        if test_emb is None or not self.cohort_embeddings:
            return self._cohort_norm(raw_score)
        top_scores = sorted([self._raw_similarity(test_emb, ce)
                             for ce in self.cohort_embeddings], reverse=True)[:SNORM_TOP_N]
        if top_scores:
            s_mean = np.mean(top_scores)
            s_std = np.std(top_scores) + 1e-6
            normalized = 1.0 / (1.0 + np.exp(-(raw_score - s_mean) / s_std * 2.0))
        else:
            normalized = raw_score
        return float(np.clip(normalized, 0.0, 1.0))

    def _cohort_norm(self, raw_score):
        if self.impostor_mean is None:
            return raw_score
        normalized = (raw_score - self.impostor_mean) / self.impostor_std
        return float(np.clip(1.0 / (1.0 + np.exp(-normalized * 2.0)), 0.0, 1.0))

    def _raw_similarity(self, emb1, emb2):
        a = np.array(emb1, dtype=np.float64)
        b = np.array(emb2, dtype=np.float64)
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def save(self, path=SCORE_NORM_MODEL_PATH):
        with open(path, 'wb') as f:
            pickle.dump({'method': self.method, 'is_trained': self.is_trained,
                         'impostor_mean': self.impostor_mean, 'impostor_std': self.impostor_std,
                         'target_mean': self.target_mean, 'target_std': self.target_std,
                         'cohort_embeddings': self.cohort_embeddings,
                         'speaker_stats': self.speaker_stats,
                         'num_speakers': self._num_speakers_at_train}, f)

    def load(self, path=SCORE_NORM_MODEL_PATH):
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.__dict__.update(state)
            self._num_speakers_at_train = state.get('num_speakers', 0)
            print(f"   ScoreNorm loaded | method={self.method}")
            return True
        except Exception as e:
            print(f"   ScoreNorm load error: {e}")
            return False


class ScoreNormManager:
    def __init__(self):
        self.normalizer = ScoreNormalizer(method=SCORE_NORM_METHOD)
        self._profiles_since_last_train = 0
        self._initialise()

    def _initialise(self):
        if not ENABLE_SCORE_NORMALIZATION:
            return
        if self.normalizer.load(SCORE_NORM_MODEL_PATH):
            return
        self.retrain()

    def retrain(self):
        speaker_embeddings = _collect_all_embeddings_from_db()
        if len(speaker_embeddings) < SCORE_NORM_MIN_SPEAKERS:
            self.normalizer.is_trained = False
            return
        self.normalizer.train(speaker_embeddings)
        self.normalizer.save(SCORE_NORM_MODEL_PATH)
        self._profiles_since_last_train = 0

    def on_new_profile(self):
        self._profiles_since_last_train += 1
        if self._profiles_since_last_train >= SCORE_NORM_RETRAIN_INTERVAL:
            self.retrain()

    @property
    def is_ready(self):
        return ENABLE_SCORE_NORMALIZATION and self.normalizer.is_trained

    def normalize(self, raw_score, test_emb=None, enroll_emb=None, speaker_id=None):
        if not self.is_ready:
            return raw_score
        return self.normalizer.normalize(raw_score, test_emb, enroll_emb, speaker_id)


# =========================================================================
# WCCN
# =========================================================================

class WCCN:
    def __init__(self, embedding_dim=PLDA_EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.W = None
        self.mean = None
        self.is_trained = False
        self._num_speakers_at_train = 0

    def train(self, speaker_embeddings):
        valid = {sid: embs for sid, embs in speaker_embeddings.items()
                 if len(embs) >= WCCN_MIN_SAMPLES_PER_SPEAKER}
        if len(valid) < WCCN_MIN_SPEAKERS:
            return
        all_embs = np.array([e for embs in valid.values() for e in embs], dtype=np.float64)
        D = all_embs.shape[1]
        self.mean = all_embs.mean(axis=0)
        Sw = np.zeros((D, D), dtype=np.float64)
        n = 0
        for embs in valid.values():
            arr = np.array(embs, dtype=np.float64)
            c = arr - arr.mean(axis=0)
            Sw += c.T @ c
            n += len(embs)
        Sw = Sw / max(n, 1) + np.eye(D) * WCCN_REGULARIZATION
        ev, evec = np.linalg.eigh(Sw)
        ev = np.maximum(ev, WCCN_REGULARIZATION)
        self.W = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
        self.is_trained = True
        self._num_speakers_at_train = len(valid)
        print(f"   WCCN trained | speakers={len(valid)}")

    def transform(self, embedding):
        if not self.is_trained or self.W is None:
            return embedding
        out = self.W @ (embedding.astype(np.float64) - self.mean)
        n = np.linalg.norm(out)
        return out / n if n > 1e-8 else out

    def save(self, path=WCCN_MODEL_PATH):
        with open(path, 'wb') as f:
            pickle.dump({'embedding_dim': self.embedding_dim, 'W': self.W,
                         'mean': self.mean, 'is_trained': self.is_trained,
                         'num_speakers': self._num_speakers_at_train}, f)

    def load(self, path=WCCN_MODEL_PATH):
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                s = pickle.load(f)
            self.embedding_dim = s['embedding_dim']
            self.W = s['W']
            self.mean = s['mean']
            self.is_trained = s['is_trained']
            self._num_speakers_at_train = s.get('num_speakers', 0)
            print(f"   WCCN loaded | speakers_at_train={self._num_speakers_at_train}")
            return True
        except Exception as e:
            print(f"   WCCN load error: {e}")
            return False


class WCCNManager:
    def __init__(self):
        self.wccn = WCCN(embedding_dim=PLDA_EMBEDDING_DIM)
        self._profiles_since_last_train = 0
        self._initialise()

    def _initialise(self):
        if not ENABLE_WCCN:
            return
        if self.wccn.load(WCCN_MODEL_PATH):
            return
        self.retrain()

    def retrain(self):
        speaker_embeddings = _collect_all_embeddings_from_db()
        if len(speaker_embeddings) < WCCN_MIN_SPEAKERS:
            self.wccn.is_trained = False
            return
        self.wccn.train(speaker_embeddings)
        self.wccn.save(WCCN_MODEL_PATH)
        self._profiles_since_last_train = 0

    def on_new_profile(self):
        self._profiles_since_last_train += 1
        if self._profiles_since_last_train >= WCCN_RETRAIN_INTERVAL:
            self.retrain()

    @property
    def is_ready(self):
        return ENABLE_WCCN and self.wccn.is_trained

    def transform(self, embedding):
        return self.wccn.transform(embedding) if self.is_ready else embedding


def _collect_all_embeddings_from_db():
    result = {}
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        rows = conn.execute('SELECT id, embedding FROM voice_profiles').fetchall()
        conn.close()
        for pid, blob in rows:
            data = pickle.loads(blob)
            embeddings = []
            if isinstance(data, dict) and 'centroids' in data:
                obj = VoiceProfileCentroid.from_dict(data)
                for c in obj.centroids:
                    emb = c.get('resemblyzer_embedding')
                    if emb is not None:
                        embeddings.append(np.array(emb, dtype=np.float64))
            elif isinstance(data, dict):
                emb = data.get('resemblyzer_embedding')
                if emb is None:
                    emb = data.get('embedding')
                if emb is not None:
                    embeddings.append(np.array(emb, dtype=np.float64))
            else:
                embeddings.append(np.array(data, dtype=np.float64))
            if embeddings:
                result[pid] = embeddings
    except Exception as e:
        print(f"   DB read error: {e}")
    return result


# =========================================================================
# MULTI-CENTROID CLUSTERING SYSTEM
# =========================================================================

class VoiceProfileCentroid:
    def __init__(self, max_centroids=MAX_CENTROIDS_PER_PROFILE):
        self.centroids = []
        self.max_centroids = max_centroids
        self.total_samples = 0
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }

    def add_embedding(self, embedding_data, quality=1.0, duration=1.0):
        self.total_samples += 1
        res_emb = self._extract_resemblyzer(embedding_data)
        ecapa_emb = embedding_data.get('ecapa_embedding') if isinstance(embedding_data, dict) else None

        if not self.centroids:
            self._create_new_centroid(res_emb, ecapa_emb, quality, duration, embedding_data)
            return

        best_idx, best_sim = self._find_best_centroid(res_emb)
        if best_sim >= CENTROID_SIMILARITY_THRESHOLD:
            self._update_centroid(best_idx, res_emb, ecapa_emb, quality, duration)
        elif len(self.centroids) < self.max_centroids:
            self._create_new_centroid(res_emb, ecapa_emb, quality, duration, embedding_data)
        else:
            self._update_centroid(best_idx, res_emb, ecapa_emb, quality, duration)

        if self.total_samples % 10 == 0:
            self._merge_similar_centroids()
            self._prune_weak_centroids()
        self.metadata['last_updated'] = datetime.now().isoformat()

    def _extract_resemblyzer(self, data):
        if isinstance(data, dict):
            emb = data.get('resemblyzer_embedding')
            if emb is None:
                emb = data.get('embedding', data)
            return np.array(emb, dtype=np.float64)
        return np.array(data, dtype=np.float64)

    def _find_best_centroid(self, embedding):
        sims = [float(np.clip(0.7 * np.dot(embedding, c['resemblyzer_embedding']), -1.0, 1.0)
                      + 0.3 / (1.0 + np.linalg.norm(embedding - c['resemblyzer_embedding'])))
                for c in self.centroids]
        idx = int(np.argmax(sims))
        return idx, sims[idx]

    def _create_new_centroid(self, res_emb, ecapa_emb, quality, duration, full_data):
        self.centroids.append({
            'resemblyzer_embedding': res_emb.copy(),
            'ecapa_embedding': ecapa_emb.copy() if ecapa_emb is not None else None,
            'weight': quality * duration, 'sample_count': 1,
            'quality_avg': quality, 'duration_total': duration,
            'gender': full_data.get('gender', 'unknown') if isinstance(full_data, dict) else 'unknown',
            'created_at': datetime.now().isoformat(),
        })

    def _update_centroid(self, idx, res_emb, ecapa_emb, quality, duration):
        c = self.centroids[idx]
        alpha = min(0.15, max(0.02, 1.0 / (c['sample_count'] + 1))) * (0.7 + 0.3 * quality)
        new_r = (1 - alpha) * c['resemblyzer_embedding'] + alpha * res_emb
        new_r /= (np.linalg.norm(new_r) + 1e-8)
        c['resemblyzer_embedding'] = new_r
        if ecapa_emb is not None and c['ecapa_embedding'] is not None:
            new_e = (1 - alpha) * c['ecapa_embedding'] + alpha * ecapa_emb
            new_e /= (np.linalg.norm(new_e) + 1e-8)
            c['ecapa_embedding'] = new_e
        elif ecapa_emb is not None:
            c['ecapa_embedding'] = ecapa_emb.copy()
        c['sample_count'] += 1
        c['weight'] += quality * duration
        c['quality_avg'] = (c['quality_avg'] * (c['sample_count'] - 1) + quality) / c['sample_count']
        c['duration_total'] += duration

    def _merge_similar_centroids(self):
        i = 0
        while i < len(self.centroids) - 1:
            j = i + 1
            while j < len(self.centroids):
                sim = float(np.clip(np.dot(
                    self.centroids[i]['resemblyzer_embedding'],
                    self.centroids[j]['resemblyzer_embedding']), -1.0, 1.0))
                if sim >= CENTROID_MERGE_THRESHOLD:
                    ci, cj = self.centroids[i], self.centroids[j]
                    tot = ci['sample_count'] + cj['sample_count']
                    wi, wj = ci['sample_count'] / tot, cj['sample_count'] / tot
                    m = wi * ci['resemblyzer_embedding'] + wj * cj['resemblyzer_embedding']
                    m /= (np.linalg.norm(m) + 1e-8)
                    ci['resemblyzer_embedding'] = m
                    if ci['ecapa_embedding'] is not None and cj['ecapa_embedding'] is not None:
                        me = wi * ci['ecapa_embedding'] + wj * cj['ecapa_embedding']
                        me /= (np.linalg.norm(me) + 1e-8)
                        ci['ecapa_embedding'] = me
                    ci['sample_count'] = tot
                    ci['weight'] += cj['weight']
                    ci['quality_avg'] = wi * ci['quality_avg'] + wj * cj['quality_avg']
                    ci['duration_total'] += cj['duration_total']
                    del self.centroids[j]
                else:
                    j += 1
            i += 1

    def _prune_weak_centroids(self):
        if len(self.centroids) <= 2:
            return
        median_count = np.median([c['sample_count'] for c in self.centroids])
        self.centroids = [c for c in self.centroids
                          if c['sample_count'] >= max(CENTROID_MIN_SAMPLES, median_count * 0.15)]

    def get_best_match_score(self, embedding_data, use_plda=False, plda_manager=None,
                             wccn_manager=None, score_norm_manager=None, speaker_id=None):
        if not self.centroids:
            return 0.0

        res_emb = self._extract_resemblyzer(embedding_data)
        if wccn_manager and wccn_manager.is_ready:
            res_emb = wccn_manager.transform(res_emb)

        ecapa_emb = embedding_data.get('ecapa_embedding') if isinstance(embedding_data, dict) else None

        scores, weights = [], []
        for c in self.centroids:
            cent_res = np.array(c['resemblyzer_embedding'], dtype=np.float64)
            if wccn_manager and wccn_manager.is_ready:
                cent_res = wccn_manager.transform(cent_res)

            cos_sim = float(np.clip(np.dot(res_emb, cent_res), -1.0, 1.0))
            euc_sim = 1.0 / (1.0 + np.linalg.norm(res_emb - cent_res))
            pear_sim = float(np.clip(np.corrcoef(res_emb, cent_res)[0, 1], -1.0, 1.0))
            res_score = 0.65 * cos_sim + 0.25 * euc_sim + 0.10 * pear_sim

            ecapa_score = None
            if ENABLE_ECAPA and ecapa_emb is not None and c.get('ecapa_embedding') is not None:
                ecapa_score = float(np.clip(np.dot(
                    np.array(ecapa_emb, dtype=np.float64),
                    np.array(c['ecapa_embedding'], dtype=np.float64)), -1.0, 1.0))

            base = (ECAPA_WEIGHT * ecapa_score + (1 - ECAPA_WEIGHT) * res_score
                    if ecapa_score is not None else res_score)

            if use_plda and plda_manager and plda_manager.is_ready:
                final = PLDA_SCORE_WEIGHT * plda_manager.score_prob(res_emb, cent_res) + (1 - PLDA_SCORE_WEIGHT) * base
            else:
                final = base

            if score_norm_manager and score_norm_manager.is_ready:
                final = score_norm_manager.normalize(final, test_emb=res_emb, enroll_emb=cent_res, speaker_id=speaker_id)

            scores.append(final)
            weights.append(c['quality_avg'] ** CENTROID_QUALITY_WEIGHT * np.log1p(c['sample_count']))

        scores = np.array(scores)
        weights = np.array(weights)
        weights /= weights.sum()
        return float(0.85 * scores.max() + 0.15 * np.average(scores, weights=weights))

    def get_primary_centroid(self):
        return max(self.centroids, key=lambda c: c['sample_count']) if self.centroids else None

    def to_dict(self):
        return {'centroids': self.centroids, 'total_samples': self.total_samples,
                'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data, max_centroids=MAX_CENTROIDS_PER_PROFILE):
        obj = cls(max_centroids=max_centroids)
        obj.centroids = data.get('centroids', [])
        obj.total_samples = data.get('total_samples', 0)
        obj.metadata = data.get('metadata', {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        })
        return obj


# =========================================================================
# ADAPTIVE THRESHOLD MANAGER
# =========================================================================

class AdaptiveThresholdManager:
    def __init__(self):
        self.similarity_threshold = BASE_SIMILARITY_THRESHOLD
        self.strong_match_threshold = BASE_STRONG_MATCH_THRESHOLD
        self.verification_threshold = BASE_VERIFICATION_THRESHOLD
        self.profile_thresholds = {}
        self._load_state()

    def _load_state(self):
        if not ENABLE_ADAPTIVE_THRESHOLDS or not os.path.exists(THRESHOLD_STATE_PATH):
            return
        try:
            with open(THRESHOLD_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
            self.similarity_threshold = state.get('similarity_threshold', BASE_SIMILARITY_THRESHOLD)
            self.strong_match_threshold = state.get('strong_match_threshold', BASE_STRONG_MATCH_THRESHOLD)
            self.verification_threshold = state.get('verification_threshold', BASE_VERIFICATION_THRESHOLD)
            self.profile_thresholds = state.get('profile_thresholds', {})
        except Exception as e:
            print(f"   Threshold load error: {e}")

    def _save_state(self):
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return
        try:
            with open(THRESHOLD_STATE_PATH, 'wb') as f:
                pickle.dump({
                    'similarity_threshold': self.similarity_threshold,
                    'strong_match_threshold': self.strong_match_threshold,
                    'verification_threshold': self.verification_threshold,
                    'profile_thresholds': self.profile_thresholds,
                }, f)
        except Exception as e:
            print(f"   Threshold save error: {e}")

    def get_threshold_for_profile(self, profile_id, total_recordings, num_centroids=1):
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return self.similarity_threshold
        base = self.profile_thresholds.get(profile_id, self.similarity_threshold)
        if total_recordings >= PROFILE_MATURITY_THRESHOLD:
            base += min((total_recordings - PROFILE_MATURITY_THRESHOLD) / 20.0, 0.08)
        if num_centroids > 2:
            base += min((num_centroids - 2) * 0.015, 0.05)
        return float(np.clip(base, THRESHOLD_MIN, THRESHOLD_MAX))

    def analyze_and_adjust(self, all_similarities, best_match_id):
        pass  # Disabled - ENABLE_ADAPTIVE_THRESHOLDS = False

    def on_match_confirmed(self, profile_id, similarity, was_correct):
        pass


threshold_manager = AdaptiveThresholdManager()


# =========================================================================
# PLDA
# =========================================================================

class PLDA:
    def __init__(self, embedding_dim=PLDA_EMBEDDING_DIM, latent_dim=PLDA_LATENT_DIM):
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.mean = None
        self.V = None
        self.Sigma_w = None
        self.is_trained = False
        self._num_speakers_at_train = 0
        self._llr_scale = 1.0

    def train(self, speaker_embeddings, n_iter=10):
        all_embs = np.array([e for embs in speaker_embeddings.values() for e in embs], dtype=np.float64)
        if all_embs.shape[0] < 2:
            return
        D = all_embs.shape[1]
        self.mean = all_embs.mean(axis=0)
        Sw = np.zeros((D, D), dtype=np.float64)
        n_total = 0
        speaker_means = {}
        for sid, embs in speaker_embeddings.items():
            arr = np.array(embs, dtype=np.float64)
            sp_mean = arr.mean(axis=0)
            speaker_means[sid] = sp_mean
            Sw += (arr - sp_mean).T @ (arr - sp_mean)
            n_total += len(embs)
        Sw = Sw / max(n_total, 1) + np.eye(D) * 1e-4
        sp_mat = np.array(list(speaker_means.values()), dtype=np.float64) - self.mean
        Sb = (sp_mat.T @ sp_mat) / max(len(speaker_means), 1)
        eigvals, eigvecs = np.linalg.eigh(Sb)
        k = min(self.latent_dim, D)
        V = eigvecs[:, np.argsort(eigvals)[::-1][:k]].T.copy()
        Sw_inv = np.linalg.inv(Sw)
        for _ in range(n_iter):
            VSwi = V @ Sw_inv
            Ainv = np.linalg.inv(np.eye(k) + VSwi @ V.T)
            Ez, EzzT = {}, {}
            for sid, embs in speaker_embeddings.items():
                arr = np.array(embs, dtype=np.float64)
                ms = arr.mean(axis=0) - self.mean
                ez = Ainv @ (VSwi @ ms)
                Ez[sid] = ez
                EzzT[sid] = Ainv + np.outer(ez, ez) * len(embs)
            nV = np.zeros((k, D), dtype=np.float64)
            dV = np.zeros((k, k), dtype=np.float64)
            Sw2 = np.zeros((D, D), dtype=np.float64)
            nt = 0
            for sid, embs in speaker_embeddings.items():
                arr = np.array(embs, dtype=np.float64)
                ni = len(embs)
                ez = Ez[sid]
                nV += ni * np.outer(ez, arr.mean(axis=0) - self.mean)
                dV += EzzT[sid]
                Sw2 += (arr - (self.mean + V.T @ ez)).T @ (arr - (self.mean + V.T @ ez))
                nt += ni
            V = np.linalg.inv(dV + np.eye(k) * 1e-6) @ nV
            Sw = Sw2 / max(nt, 1) + np.eye(D) * 1e-4
            Sw_inv = np.linalg.inv(Sw)
        self.V = V
        self.Sigma_w = Sw
        self.is_trained = True
        self._num_speakers_at_train = len(speaker_embeddings)
        self._llr_scale = self._calibrate_scale(speaker_embeddings)
        print(f"   PLDA trained | speakers={len(speaker_embeddings)} | latent={k}")

    def _calibrate_scale(self, speaker_embeddings, n=30):
        sids = list(speaker_embeddings.keys())
        if len(sids) < 2:
            return 1.0
        rng = np.random.default_rng(0)
        llrs = []
        for _ in range(n):
            s = rng.choice(sids)
            d = s
            while d == s:
                d = rng.choice(sids)
            es = speaker_embeddings[s]
            i, j = rng.choice(len(es), 2, replace=(len(es) < 2))
            llrs.append(abs(self._raw_score(es[i], es[j])))
            llrs.append(abs(self._raw_score(es[0], speaker_embeddings[d][0])))
        return max(float(np.median(llrs)), 1e-6)

    def score(self, e1, e2):
        return self._raw_score(e1, e2) / self._llr_scale

    def _raw_score(self, e1, e2):
        if not self.is_trained:
            return 0.0
        x1 = e1.astype(np.float64) - self.mean
        x2 = e2.astype(np.float64) - self.mean
        D = self.embedding_dim
        Sw_inv = np.linalg.inv(self.Sigma_w)
        Ss = self.Sigma_w + self.V.T @ self.V
        Ss_inv = np.linalg.inv(Ss + np.eye(D) * 1e-6)
        _, ldw = np.linalg.slogdet(self.Sigma_w)
        _, lds = np.linalg.slogdet(Ss)
        qd = (x1 @ Sw_inv @ x1 + x2 @ Sw_inv @ x2) * 0.5
        qs = ((x1 - x2) @ Sw_inv @ (x1 - x2) + (x1 + x2) @ Ss_inv @ (x1 + x2)) * 0.25
        return float((ldw - lds) + (qd - qs))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x)) if x >= 0 else np.exp(x) / (1.0 + np.exp(x))

    def score_normalised(self, e1, e2):
        return self.sigmoid(self.score(e1, e2))

    def save(self, path=PLDA_MODEL_PATH):
        with open(path, 'wb') as f:
            pickle.dump({'embedding_dim': self.embedding_dim, 'latent_dim': self.latent_dim,
                         'mean': self.mean, 'V': self.V, 'Sigma_w': self.Sigma_w,
                         'is_trained': self.is_trained,
                         'num_speakers': self._num_speakers_at_train,
                         'llr_scale': self._llr_scale}, f)

    def load(self, path=PLDA_MODEL_PATH):
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                s = pickle.load(f)
            self.__dict__.update(s)
            self._llr_scale = s.get('llr_scale', 1.0)
            print(f"   PLDA loaded | speakers_at_train={s.get('num_speakers', 0)}")
            return True
        except Exception as e:
            print(f"   PLDA load error: {e}")
            return False


class PLDAManager:
    def __init__(self):
        self.plda = PLDA()
        self._profiles_since_last_train = 0
        self._initialise()

    def _initialise(self):
        if not ENABLE_PLDA:
            return
        if self.plda.load(PLDA_MODEL_PATH):
            return
        self.retrain()

    def retrain(self):
        embs = _collect_all_embeddings_from_db()
        if len(embs) < PLDA_MIN_SPEAKERS:
            self.plda.is_trained = False
            return
        self.plda.train(embs)
        self.plda.save(PLDA_MODEL_PATH)
        self._profiles_since_last_train = 0

    def on_new_profile(self):
        self._profiles_since_last_train += 1
        if self._profiles_since_last_train >= PLDA_RETRAIN_INTERVAL:
            self.retrain()

    @property
    def is_ready(self):
        return ENABLE_PLDA and self.plda.is_trained

    def score_llr(self, e1, e2):
        return self.plda.score(e1, e2) if self.is_ready else 0.0

    def score_prob(self, e1, e2):
        return self.plda.score_normalised(e1, e2) if self.is_ready else 0.5


# =========================================================================
# INITIALIZATION
# =========================================================================

print("Loading VocalD Voice Recognition System ...")

encoder = None
try:
    encoder = VoiceEncoder()
    print("Resemblyzer encoder loaded")
except Exception as e:
    print(f"Resemblyzer error: {e}")

ecapa_classifier = None
if ENABLE_ECAPA and EncoderClassifier is not None:
    try:
        print("Loading ECAPA-TDNN ...")
        ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"})
        print("ECAPA-TDNN loaded")
    except Exception as e:
        print(f"ECAPA-TDNN failed: {e}")
        ecapa_classifier = None
        ENABLE_ECAPA = False
else:
    ENABLE_ECAPA = False

diarization_pipeline = None
try:
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("No HUGGINGFACE_TOKEN set")
    else:
        import torch.serialization
        try:
            from torch.torch_version import TorchVersion
            torch.serialization.add_safe_globals([TorchVersion])
            print("Registered TorchVersion as safe global")
        except Exception as _tv_exc:
            print(f"Could not register TorchVersion: {_tv_exc}")

        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diarization_pipeline.to(torch.device("cpu"))
        num_cores = multiprocessing.cpu_count()
        torch.set_num_threads(num_cores)
        torch.set_grad_enabled(False)
        print(f"PyAnnote 3.1 loaded ({num_cores} cores)")
except Exception as e:
    print(f"PyAnnote error: {e}")

vad = None
if ENABLE_VAD:
    try:
        vad = webrtcvad.Vad(3)
        print("VAD enabled")
    except Exception:
        ENABLE_VAD = False

print("Initialising PLDA ...")
plda_manager = PLDAManager()

print("Initialising WCCN ...")
wccn_manager = WCCNManager()

print("Initialising Score Normalizer ...")
score_norm_manager = ScoreNormManager()

print("=" * 60)
print(f"  Resemblyzer   : {'OK' if encoder else 'MISSING'}")
print(f"  ECAPA-TDNN    : {'OK' if ecapa_classifier else 'disabled'}")
print(f"  Diarization   : {'OK' if diarization_pipeline else 'MISSING'}")
print(f"  PLDA          : {'trained' if plda_manager.is_ready else 'waiting for data'}")
print(f"  WCCN          : {'trained' if wccn_manager.is_ready else 'waiting for data'}")
print(f"  ScoreNorm     : {'trained' if score_norm_manager.is_ready else 'waiting for data'}")
print(f"  Threshold     : {threshold_manager.similarity_threshold:.3f}")
print("=" * 60)


# =========================================================================
# AUDIO PROCESSING FUNCTIONS
# =========================================================================

def _convert_to_wav(audio_path):
    """Convert any audio to WAV using ffmpeg for consistent processing."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == '.wav':
        return audio_path, False
    try:
        import subprocess
        fd, wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        result = subprocess.run(
            ['ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', '-y', wav_path],
            capture_output=True, timeout=60)
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path, True
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return audio_path, False
    except Exception as e:
        print(f"ffmpeg conversion failed: {e}")
        return audio_path, False


def apply_vad(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_VAD or vad is None:
        return wav
    try:
        wav_int16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        frame_length = int(sample_rate * 30 / 1000)
        speech_frames = []
        for i in range(0, len(wav_int16) - frame_length, frame_length):
            if vad.is_speech(wav_int16[i:i + frame_length].tobytes(), sample_rate):
                speech_frames.append(wav[i:i + frame_length])
        return np.concatenate(speech_frames) if speech_frames else wav
    except Exception as e:
        print(f"VAD error: {e}")
        return wav


def advanced_noise_filter(wav, sample_rate=SAMPLE_RATE):
    try:
        wav = wav - np.mean(wav)
        mx = np.max(np.abs(wav))
        if mx > 0:
            wav = wav / mx
        if not ENABLE_NOISE_FILTERING:
            return wav
        nyquist = sample_rate / 2.0
        lo = LOW_FREQ_CUTOFF / nyquist
        if 0 < lo < 1.0:
            b, a = butter(5, lo, btype='high')
            wav = filtfilt(b, a, wav)
        hi = HIGH_FREQ_CUTOFF / nyquist
        if 0 < hi < 1.0:
            b, a = butter(5, hi, btype='low')
            wav = filtfilt(b, a, wav)
        wav = gaussian_filter1d(wav, sigma=0.8)
        mx = np.max(np.abs(wav))
        return wav / mx if mx > 0 else wav
    except Exception as e:
        print(f"Filtering error: {e}")
        wav = wav - np.mean(wav)
        mx = np.max(np.abs(wav))
        return wav / mx if mx > 0 else wav


def calculate_audio_quality(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_QUALITY_SCORING or len(wav) < sample_rate:
        return 1.0
    try:
        nn = min(int(0.2 * sample_rate), len(wav) // 4)
        noise_pwr = np.mean(wav[:nn] ** 2) + 1e-10
        sig_pwr = np.mean(wav[nn:] ** 2) + 1e-10
        if noise_pwr > sig_pwr * 0.5:
            snr_score = float(np.clip(np.std(wav) / 0.10, 0, 1))
        else:
            snr = 10 * np.log10(sig_pwr / noise_pwr)
            snr_score = float(np.clip((snr - 5) / 20, 0, 1))
        dr_score = float(np.clip(np.std(wav) / 0.15, 0, 1))
        clip_score = 1.0 - float(np.clip(np.sum(np.abs(wav) > 0.95) / len(wav) * 50, 0, 1))
        zcr_score = float(np.clip(np.sum(np.abs(np.diff(np.sign(wav)))) / len(wav) / 0.1, 0, 1))
        return 0.40 * snr_score + 0.25 * dr_score + 0.25 * clip_score + 0.10 * zcr_score
    except Exception:
        return 0.5


def classify_gender(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_GENDER_CLASSIFICATION or len(wav) < sample_rate:
        return 'unknown'
    try:
        f0 = librosa.yin(wav, fmin=75, fmax=400, sr=sample_rate)
        f0_v = f0[f0 > 75]
        if len(f0_v) < 10:
            return 'unknown'
        med = np.median(f0_v)
        return 'male' if med < 145 else ('female' if med > 165 else 'unknown')
    except Exception:
        return 'unknown'


def extract_ecapa_embedding(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_ECAPA or ecapa_classifier is None:
        return None
    try:
        if sample_rate != 16000:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
        with torch.no_grad():
            emb = ecapa_classifier.encode_batch(
                torch.FloatTensor(wav).unsqueeze(0)).squeeze().cpu().numpy()
        n = np.linalg.norm(emb)
        return emb / (n + 1e-8) if n > 1e-8 else emb
    except Exception as e:
        print(f"ECAPA error: {e}")
        return None


def extract_voice_embedding(audio_path, start_time=None, end_time=None):
    if encoder is None:
        return None
    try:
        # Convert to WAV first for consistent processing
        wav_path, converted = _convert_to_wav(audio_path)
        try:
            wav = preprocess_wav(Path(wav_path))
        finally:
            if converted and os.path.exists(wav_path):
                os.remove(wav_path)

        if start_time is not None and end_time is not None:
            s = max(0, int(start_time * SAMPLE_RATE))
            e = min(len(wav), int(end_time * SAMPLE_RATE))
            if s >= e:
                return None
            wav = wav[s:e]
            if len(wav) < int(MIN_SEGMENT_DURATION * SAMPLE_RATE):
                return None

        if ENABLE_VAD:
            wav = apply_vad(wav)
            if len(wav) < int(MIN_SEGMENT_DURATION * SAMPLE_RATE):
                return None

        wav_clean = advanced_noise_filter(wav)
        quality = calculate_audio_quality(wav_clean)
        gender = classify_gender(wav_clean)

        res_emb = encoder.embed_utterance(wav_clean)
        res_emb = res_emb / (np.linalg.norm(res_emb) + 1e-8)
        ecapa_emb = extract_ecapa_embedding(wav_clean, SAMPLE_RATE)

        return {
            'resemblyzer_embedding': res_emb,
            'ecapa_embedding': ecapa_emb,
            'quality': quality,
            'gender': gender,
            'duration': len(wav) / SAMPLE_RATE,
        }
    except Exception as e:
        print(f"Embedding extraction error: {e}")
        return None


def perform_smart_diarization(audio_path):
    if diarization_pipeline is None:
        return None
    tmp = None
    try:
        print(f"Diarizing: {Path(audio_path).name}")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(y) / sr
        print(f"Duration: {duration:.1f}s")

        fd, tmp = tempfile.mkstemp(prefix='vocald_diar_', suffix='.wav')
        os.close(fd)
        sf.write(tmp, y, SAMPLE_RATE, subtype='PCM_16')

        with torch.no_grad():
            print("  Strategy 1: unconstrained detection ...")
            out = diarization_pipeline(tmp)

        # FIX: PyAnnote 3.1 returns Annotation directly
        speakers_data = {}
        for segment, _, label in out.itertracks(yield_label=True):
            speakers_data.setdefault(label, []).append({
                'start': segment.start, 'end': segment.end,
                'duration': segment.end - segment.start,
            })
        print(f"  Strategy 1: {len(speakers_data)} speaker(s)")

        needs_retry = (
            (len(speakers_data) == 1 and duration > 15) or
            (duration > 180 and len(speakers_data) < 3)
        )
        if needs_retry:
            print("  Strategy 2: forcing min_speakers=2 ...")
            with torch.no_grad():
                out = diarization_pipeline(tmp, min_speakers=2, max_speakers=20)
            speakers_data = {}
            for segment, _, label in out.itertracks(yield_label=True):
                speakers_data.setdefault(label, []).append({
                    'start': segment.start, 'end': segment.end,
                    'duration': segment.end - segment.start,
                })
            print(f"  Strategy 2: {len(speakers_data)} speaker(s)")

        filtered = {
            lbl: {'segments': segs, 'total_duration': sum(s['duration'] for s in segs)}
            for lbl, segs in speakers_data.items()
            if sum(s['duration'] for s in segs) >= MIN_SPEAKING_TIME
        }
        print(f"After duration filter: {len(filtered)} speaker(s)")
        return filtered or None

    except Exception as e:
        print(f"Diarization error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# =========================================================================
# SIMILARITY & MATCHING
# =========================================================================

def _extract_resemblyzer(data):
    if isinstance(data, dict):
        emb = data.get('resemblyzer_embedding')
        if emb is None:
            emb = data.get('embedding', data)
        return np.array(emb, dtype=np.float64)
    return np.array(data, dtype=np.float64)


def find_matching_speaker(embedding_data):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        profiles = conn.execute(
            'SELECT id, name, embedding, total_recordings FROM voice_profiles').fetchall()
        conn.close()
        if not profiles:
            return None

        all_sims = []
        best_match, best_sim = None, 0.0

        for pid, name, blob, total_recordings in profiles:
            stored = pickle.loads(blob)
            if isinstance(stored, dict) and 'centroids' in stored:
                obj = VoiceProfileCentroid.from_dict(stored)
                sim = obj.get_best_match_score(
                    embedding_data, use_plda=True,
                    plda_manager=plda_manager,
                    wccn_manager=wccn_manager,
                    score_norm_manager=score_norm_manager,
                    speaker_id=pid)
                num_centroids = len(obj.centroids)
            else:
                sim = _calculate_similarity_legacy(embedding_data, stored)
                num_centroids = 1

            all_sims.append((sim, pid, name))
            threshold = threshold_manager.get_threshold_for_profile(pid, total_recordings, num_centroids)

            if sim > threshold and sim > best_sim:
                verified, vsim = _verify_speaker_match(embedding_data, pid)
                if verified:
                    best_sim = sim
                    best_match = {
                        'id': pid, 'name': name,
                        'confidence': sim * 100,
                        'similarity': sim, 'verified': True,
                        'num_centroids': num_centroids,
                    }

        threshold_manager.analyze_and_adjust(all_sims, best_match['id'] if best_match else None)
        return best_match

    except Exception as e:
        print(f"Error finding match: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_similarity_legacy(emb1_data, emb2_data):
    r1 = _extract_resemblyzer(emb1_data)
    r2 = _extract_resemblyzer(emb2_data)
    if wccn_manager.is_ready:
        r1 = wccn_manager.transform(r1)
        r2 = wccn_manager.transform(r2)
    cos = float(np.clip(np.dot(r1, r2), -1.0, 1.0))
    euc = 1.0 / (1.0 + np.linalg.norm(r1 - r2))
    pear = float(np.clip(np.corrcoef(r1, r2)[0, 1], -1.0, 1.0)) if len(r1) > 1 else cos
    res_score = 0.65 * cos + 0.25 * euc + 0.10 * pear
    ecapa_score = None
    if ENABLE_ECAPA and isinstance(emb1_data, dict) and isinstance(emb2_data, dict):
        e1 = emb1_data.get('ecapa_embedding')
        e2 = emb2_data.get('ecapa_embedding')
        if e1 is not None and e2 is not None:
            ecapa_score = float(np.clip(np.dot(
                np.array(e1, dtype=np.float64), np.array(e2, dtype=np.float64)), -1.0, 1.0))
    base = (ECAPA_WEIGHT * ecapa_score + (1 - ECAPA_WEIGHT) * res_score
            if ecapa_score is not None else res_score)
    if plda_manager.is_ready:
        final = PLDA_SCORE_WEIGHT * plda_manager.score_prob(r1, r2) + (1 - PLDA_SCORE_WEIGHT) * base
    else:
        final = base
    if score_norm_manager.is_ready:
        final = score_norm_manager.normalize(final, test_emb=r1)
    if isinstance(emb1_data, dict) and isinstance(emb2_data, dict):
        g1 = emb1_data.get('gender', 'unknown')
        g2 = emb2_data.get('gender', 'unknown')
        if g1 != 'unknown' and g2 != 'unknown' and g1 != g2:
            final *= 0.85
    return float(final)


def _verify_speaker_match(embedding_data, profile_id):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        row = conn.execute(
            'SELECT embedding, total_recordings FROM voice_profiles WHERE id=?',
            (profile_id,)).fetchone()
        conn.close()
        if not row:
            return False, 0.0

        stored = pickle.loads(row[0])
        total = row[1]

        if isinstance(stored, dict) and 'centroids' in stored:
            obj = VoiceProfileCentroid.from_dict(stored)
            sim = obj.get_best_match_score(
                embedding_data, use_plda=True,
                plda_manager=plda_manager,
                wccn_manager=wccn_manager,
                score_norm_manager=score_norm_manager,
                speaker_id=profile_id)
            num_centroids = len(obj.centroids)
            primary = obj.get_primary_centroid()
        else:
            sim = _calculate_similarity_legacy(embedding_data, stored)
            num_centroids = 1
            primary = stored

        threshold = threshold_manager.get_threshold_for_profile(profile_id, total, num_centroids)
        r_new = _extract_resemblyzer(embedding_data)
        r_old = _extract_resemblyzer(primary)
        if wccn_manager.is_ready:
            r_new = wccn_manager.transform(r_new)
            r_old = wccn_manager.transform(r_old)

        cos_ok = float(np.clip(np.dot(r_new, r_old), -1.0, 1.0)) >= 0.72
        euc_ok = (1.0 / (1.0 + np.linalg.norm(r_new - r_old))) >= 0.65
        plda_ok = (plda_manager.score_llr(r_new, r_old) >= 0.0) if plda_manager.is_ready else True

        return sim >= threshold and cos_ok and euc_ok and plda_ok, sim

    except Exception as e:
        print(f"Verification error: {e}")
        return False, 0.0


# =========================================================================
# DATABASE OPERATIONS
# =========================================================================

def create_voice_profile(name, embedding_data):
    try:
        now = datetime.now().isoformat()
        if ENABLE_MULTI_CENTROID:
            obj = VoiceProfileCentroid(max_centroids=MAX_CENTROIDS_PER_PROFILE)
            quality = embedding_data.get('quality', 1.0) if isinstance(embedding_data, dict) else 1.0
            duration = embedding_data.get('duration', 1.0) if isinstance(embedding_data, dict) else 1.0
            obj.add_embedding(embedding_data, quality, duration)
            stored_data = obj.to_dict()
        else:
            stored_data = embedding_data

        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.execute(
            'INSERT INTO voice_profiles (name, embedding, first_seen, last_seen, total_recordings) '
            'VALUES (?, ?, ?, ?, ?)',
            (name, pickle.dumps(stored_data), now, now, 1))
        pid = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.commit()
        conn.close()

        plda_manager.on_new_profile()
        wccn_manager.on_new_profile()
        score_norm_manager.on_new_profile()
        print(f"Created profile | id={pid} name='{name}'")
        return pid
    except Exception as e:
        print(f"Error creating profile: {e}")
        return None


def update_voice_profile(profile_id, new_embedding_data):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        row = conn.execute(
            'SELECT embedding, total_recordings FROM voice_profiles WHERE id=?',
            (profile_id,)).fetchone()
        if not row:
            conn.close()
            return
        old_data = pickle.loads(row[0])
        total = row[1]
        quality = new_embedding_data.get('quality', 1.0) if isinstance(new_embedding_data, dict) else 1.0
        duration = new_embedding_data.get('duration', 1.0) if isinstance(new_embedding_data, dict) else 1.0

        if ENABLE_MULTI_CENTROID:
            if isinstance(old_data, dict) and 'centroids' in old_data:
                obj = VoiceProfileCentroid.from_dict(old_data)
            else:
                obj = VoiceProfileCentroid()
                obj.add_embedding(old_data, 1.0, 1.0)
            obj.add_embedding(new_embedding_data, quality, duration)
            refined = obj.to_dict()
        else:
            alpha = max(ADAPTIVE_LEARNING_RATE[0],
                        min(1.0 / (total + 1), ADAPTIVE_LEARNING_RATE[1])) * quality
            old_r = _extract_resemblyzer(old_data)
            new_r = _extract_resemblyzer(new_embedding_data)
            refined_r = (1 - alpha) * old_r + alpha * new_r
            refined_r /= (np.linalg.norm(refined_r) + 1e-8)
            refined = {'resemblyzer_embedding': refined_r, 'quality': quality,
                       'gender': new_embedding_data.get('gender', 'unknown') if isinstance(new_embedding_data, dict) else 'unknown'}

        conn.execute(
            'UPDATE voice_profiles SET last_seen=?, total_recordings=total_recordings+1, embedding=? WHERE id=?',
            (datetime.now().isoformat(), pickle.dumps(refined), profile_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating profile: {e}")


# =========================================================================
# SEGMENT PROCESSING
# =========================================================================

def merge_close_segments(segments, gap=0.15):
    if not segments:
        return segments
    segs = sorted(segments, key=lambda x: x['start'])
    merged = [segs[0].copy()]
    for s in segs[1:]:
        if s['start'] - merged[-1]['end'] <= gap:
            merged[-1]['end'] = s['end']
            merged[-1]['duration'] = merged[-1]['end'] - merged[-1]['start']
        else:
            merged.append(s.copy())
    return merged


def process_speaker_parallel(args):
    speaker_label, speaker_info, audio_path = args
    total_duration = speaker_info['total_duration']
    merged = merge_close_segments(speaker_info['segments'])
    top_segs = sorted(merged, key=lambda x: x['duration'], reverse=True)[:MAX_SEGMENTS_PER_SPEAKER]

    embeddings_data, weights = [], []
    for seg in top_segs:
        if seg['duration'] < MIN_SEGMENT_DURATION:
            continue
        emb = extract_voice_embedding(audio_path, seg['start'], seg['end'])
        if emb:
            embeddings_data.append(emb)
            weights.append(seg['duration'] * emb.get('quality', 1.0))

    if not embeddings_data:
        return None

    if len(embeddings_data) > 3:
        vecs = np.array([e['resemblyzer_embedding'] for e in embeddings_data])
        med = np.median(vecs, axis=0)
        dists = [np.linalg.norm(v - med) for v in vecs]
        med_d = np.median(dists)
        keep = [(e, w) for e, w, d in zip(embeddings_data, weights, dists)
                if d < med_d * OUTLIER_REJECTION_FACTOR]
        if keep:
            embeddings_data, weights = zip(*keep)
            embeddings_data, weights = list(embeddings_data), list(weights)

    w_total = sum(weights)
    w_norm = [w / w_total for w in weights]
    avg_r = np.average([e['resemblyzer_embedding'] for e in embeddings_data], axis=0, weights=w_norm)
    avg_r /= (np.linalg.norm(avg_r) + 1e-8)

    avg_e = None
    if ENABLE_ECAPA:
        pairs = [(e['ecapa_embedding'], wn) for e, wn in zip(embeddings_data, w_norm)
                 if e.get('ecapa_embedding') is not None]
        if pairs:
            vecs, ws = zip(*pairs)
            ws_arr = np.array(ws); ws_arr /= ws_arr.sum()
            avg_e = np.average(list(vecs), axis=0, weights=ws_arr)
            avg_e /= (np.linalg.norm(avg_e) + 1e-8)

    genders = [e.get('gender', 'unknown') for e in embeddings_data]
    return {
        'label': speaker_label,
        'embedding': {
            'resemblyzer_embedding': avg_r,
            'ecapa_embedding': avg_e,
            'quality': float(np.mean([e.get('quality', 1.0) for e in embeddings_data])),
            'gender': max(set(genders), key=genders.count),
            'duration': total_duration,
        },
        'duration': total_duration,
        'num_embeddings': len(embeddings_data),
    }


# =========================================================================
# MAIN PROCESSING
# =========================================================================

def process_audio_file(audio_path, filename):
    try:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {filename}")
        print(f"{'='*60}")
        t0 = time.time()
        stem = Path(filename).stem
        speakers_data = perform_smart_diarization(audio_path)
        speakers = []

        if not speakers_data:
            print("Single-speaker mode")
            emb_data = extract_voice_embedding(audio_path)
            if emb_data is None:
                return [{'speaker_index': 0, 'name': f"Unknown ({stem})",
                         'confidence': 0.0, 'voice_profile_id': None}]
            if emb_data.get('duration', 0) < MIN_PROFILE_DURATION:
                return [{'speaker_index': 0, 'name': f"Too Short ({stem})",
                         'confidence': 0.0, 'voice_profile_id': None}]
            match = find_matching_speaker(emb_data)
            if match:
                print(f"MATCHED: {match['name']} ({match['confidence']:.1f}%)")
                speakers.append({'speaker_index': 0, 'name': match['name'],
                                 'confidence': round(match['confidence'], 1),
                                 'voice_profile_id': match['id']})
                if match.get('verified') and match['similarity'] >= threshold_manager.strong_match_threshold:
                    update_voice_profile(match['id'], emb_data)
            else:
                name = f"Speaker {stem}"
                pid = create_voice_profile(name, emb_data)
                print(f"NEW profile: '{name}' id={pid}")
                speakers.append({'speaker_index': 0, 'name': name,
                                 'confidence': 95.0, 'voice_profile_id': pid})

        else:
            print(f"Multi-speaker: {len(speakers_data)} speaker(s)")
            args = [(lbl, info, audio_path)
                    for lbl, info in sorted(speakers_data.items(),
                                            key=lambda x: x[1]['total_duration'], reverse=True)]
            max_w = min(len(args), multiprocessing.cpu_count())
            with ThreadPoolExecutor(max_workers=max_w) as ex:
                future_map = {ex.submit(process_speaker_parallel, a): a[0] for a in args}
                results = [f.result() for f in as_completed(future_map)]

            results = sorted([r for r in results if r], key=lambda x: x['duration'], reverse=True)
            idx = 0
            for result in results:
                emb_data = result['embedding']
                if result['duration'] < MIN_PROFILE_DURATION:
                    print(f"  {result['label']}: {result['duration']:.1f}s - too short, skipping")
                    continue
                print(f"  {result['label']}: {result['duration']:.1f}s "
                      f"({result['num_embeddings']} segs) "
                      f"gender={emb_data.get('gender','?')} "
                      f"q={emb_data.get('quality', 0)*100:.0f}%")
                match = find_matching_speaker(emb_data)
                if match:
                    print(f"  -> MATCHED: {match['name']} ({match['confidence']:.1f}%)")
                    speakers.append({'speaker_index': idx, 'name': match['name'],
                                     'confidence': round(match['confidence'], 1),
                                     'voice_profile_id': match['id']})
                    if match.get('verified') and match['similarity'] >= threshold_manager.strong_match_threshold:
                        update_voice_profile(match['id'], emb_data)
                else:
                    name = f"Speaker {idx + 1}"
                    pid = create_voice_profile(name, emb_data)
                    print(f"  -> NEW: '{name}' id={pid}")
                    speakers.append({'speaker_index': idx, 'name': name,
                                     'confidence': 95.0, 'voice_profile_id': pid})
                idx += 1

        print(f"\nCOMPLETED: {len(speakers)} speaker(s) in {time.time() - t0:.1f}s")
        return speakers

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return [{'speaker_index': 0, 'name': f"Error ({Path(filename).stem})",
                 'confidence': 0.0, 'voice_profile_id': None}]


if __name__ == "__main__":
    print("VocalD ready.")
    print(f"Threshold: {threshold_manager.similarity_threshold:.3f}")