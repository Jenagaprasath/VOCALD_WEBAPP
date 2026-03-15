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
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import librosa
import soundfile as sf
import webrtcvad

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
MIN_PROFILE_DURATION = 3.0

# BASE SIMILARITY THRESHOLDS
BASE_SIMILARITY_THRESHOLD = 0.68
BASE_STRONG_MATCH_THRESHOLD = 0.78
BASE_VERIFICATION_THRESHOLD = 0.75
ADAPTIVE_LEARNING_RATE = (0.02, 0.10)

# CENTROID CLUSTERING SETTINGS
ENABLE_MULTI_CENTROID = True
MAX_CENTROIDS_PER_PROFILE = 5
CENTROID_SIMILARITY_THRESHOLD = 0.85
CENTROID_MIN_SAMPLES = 2
CENTROID_MERGE_THRESHOLD = 0.92
CENTROID_QUALITY_WEIGHT = 0.3

# ADAPTIVE THRESHOLD SETTINGS
ENABLE_ADAPTIVE_THRESHOLDS = True
THRESHOLD_MIN = 0.60
THRESHOLD_MAX = 0.88
THRESHOLD_STEP = 0.02
PROFILE_MATURITY_THRESHOLD = 5
CONFIDENCE_PERCENTILE = 75
ADAPTIVE_WINDOW_SIZE = 50
FALSE_ACCEPT_PENALTY = 0.04
FALSE_REJECT_REWARD = -0.02
SIMILARITY_GAP_THRESHOLD = 0.15

# ECAPA-TDNN
ENABLE_ECAPA = True
ECAPA_WEIGHT = 0.60

# PLDA SETTINGS
ENABLE_PLDA = True
PLDA_MIN_SPEAKERS = 3
PLDA_EMBEDDING_DIM = 256
PLDA_LATENT_DIM = 64
PLDA_RETRAIN_INTERVAL = 5
PLDA_SCORE_WEIGHT = 0.70
PLDA_LLR_THRESHOLD = 1.5

# WCCN SETTINGS
ENABLE_WCCN = True
WCCN_MIN_SPEAKERS = 5
WCCN_MIN_SAMPLES_PER_SPEAKER = 2
WCCN_RETRAIN_INTERVAL = 10
WCCN_REGULARIZATION = 1e-4

# SCORE NORMALIZATION SETTINGS
ENABLE_SCORE_NORMALIZATION = True
SCORE_NORM_METHOD = 'adaptive_snorm'  # 'zt_norm', 'adaptive_snorm', 'cohort'
COHORT_SIZE = 100
ZT_NORM_SIZE = 50
SNORM_TOP_N = 20
SCORE_NORM_MIN_SPEAKERS = 8
SCORE_NORM_RETRAIN_INTERVAL = 15

# ADVANCED FEATURES
ENABLE_VAD = True
ENABLE_GENDER_CLASSIFICATION = True
ENABLE_QUALITY_SCORING = True
ENABLE_POST_CLUSTERING = True
OUTLIER_REJECTION_FACTOR = 2.0

# NOISE FILTERING
ENABLE_NOISE_FILTERING = True
LOW_FREQ_CUTOFF = 85
HIGH_FREQ_CUTOFF = 8000


# =========================================================================
# SCORE NORMALIZATION SYSTEM
# =========================================================================

class ScoreNormalizer:
    """
    Advanced score normalization for speaker verification.
    Implements ZT-Norm, Adaptive S-Norm, and Cohort-based normalization.
    """
    
    def __init__(self, method='adaptive_snorm'):
        self.method = method
        self.is_trained = False
        
        # Cohort statistics
        self.impostor_mean = None
        self.impostor_std = None
        self.target_mean = None
        self.target_std = None
        
        # Cohort embeddings
        self.cohort_embeddings = []
        self.speaker_stats = {}
        
        self._num_speakers_at_train = 0
    
    def train(self, speaker_embeddings: dict[int, list[np.ndarray]]) -> None:
        """Train score normalization from speaker embeddings."""
        if len(speaker_embeddings) < SCORE_NORM_MIN_SPEAKERS:
            print(f"   ℹ️  ScoreNorm: need ≥ {SCORE_NORM_MIN_SPEAKERS} speakers. "
                  f"Current: {len(speaker_embeddings)}")
            return
        
        # Build universal cohort
        all_embeddings = []
        for sid, embs in speaker_embeddings.items():
            all_embeddings.extend(embs)
        
        if len(all_embeddings) > COHORT_SIZE:
            indices = np.random.choice(len(all_embeddings), COHORT_SIZE, replace=False)
            self.cohort_embeddings = [all_embeddings[i] for i in indices]
        else:
            self.cohort_embeddings = all_embeddings.copy()
        
        # Compute impostor score statistics
        impostor_scores = []
        target_scores = []
        
        speaker_ids = list(speaker_embeddings.keys())
        
        # Sample impostor pairs
        rng = np.random.default_rng()
        for _ in range(min(500, len(speaker_ids) * 10)):
            sid1, sid2 = rng.choice(speaker_ids, 2, replace=False)
            embs1 = speaker_embeddings[sid1]
            embs2 = speaker_embeddings[sid2]
            emb1 = embs1[rng.integers(0, len(embs1))]
            emb2 = embs2[rng.integers(0, len(embs2))]
            score = self._raw_similarity(emb1, emb2)
            impostor_scores.append(score)
        
        # Sample target pairs
        for sid in speaker_ids:
            embs = speaker_embeddings[sid]
            if len(embs) >= 2:
                for _ in range(min(5, len(embs))):
                    idx1, idx2 = rng.choice(len(embs), 2, replace=False)
                    emb1 = embs[idx1]
                    emb2 = embs[idx2]
                    score = self._raw_similarity(emb1, emb2)
                    target_scores.append(score)
        
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
        
        # Per-speaker statistics for ZT-norm
        for sid, embs in speaker_embeddings.items():
            if len(embs) < 2:
                continue
            
            cohort_scores = []
            for emb in embs[:min(5, len(embs))]:
                for cohort_emb in self.cohort_embeddings[:ZT_NORM_SIZE]:
                    cohort_scores.append(self._raw_similarity(emb, cohort_emb))
            
            if cohort_scores:
                self.speaker_stats[sid] = {
                    'mean': float(np.mean(cohort_scores)),
                    'std': float(np.std(cohort_scores) + 1e-6)
                }
        
        self.is_trained = True
        self._num_speakers_at_train = len(speaker_embeddings)
        
        print(f"   ✅ ScoreNorm trained  |  method={self.method}  "
              f"|  speakers={len(speaker_embeddings)}  "
              f"|  cohort_size={len(self.cohort_embeddings)}")
        print(f"      Stats: impostor={self.impostor_mean:.3f}±{self.impostor_std:.3f}  "
              f"target={self.target_mean:.3f}±{self.target_std:.3f}")
    
    def normalize(self, raw_score: float, test_emb: np.ndarray = None, 
                  enroll_emb: np.ndarray = None, speaker_id: int = None) -> float:
        """Normalize a raw similarity score."""
        if not self.is_trained:
            return raw_score
        
        if self.method == 'zt_norm':
            return self._zt_norm(raw_score, test_emb, enroll_emb, speaker_id)
        elif self.method == 'adaptive_snorm':
            return self._adaptive_snorm(raw_score, test_emb)
        elif self.method == 'cohort':
            return self._cohort_norm(raw_score)
        else:
            return raw_score
    
    def _zt_norm(self, raw_score: float, test_emb: np.ndarray, 
                 enroll_emb: np.ndarray, speaker_id: int) -> float:
        """ZT-Norm: Z-norm + T-norm normalization."""
        # Z-norm
        z_scores = []
        if test_emb is not None and len(self.cohort_embeddings) > 0:
            for cohort_emb in self.cohort_embeddings[:ZT_NORM_SIZE]:
                z_scores.append(self._raw_similarity(test_emb, cohort_emb))
        
        if z_scores:
            z_mean = np.mean(z_scores)
            z_std = np.std(z_scores) + 1e-6
            z_normalized = (raw_score - z_mean) / z_std
        else:
            z_normalized = raw_score
        
        # T-norm
        t_normalized = z_normalized
        if enroll_emb is not None and len(self.cohort_embeddings) > 0:
            t_scores = []
            for cohort_emb in self.cohort_embeddings[:ZT_NORM_SIZE]:
                t_scores.append(self._raw_similarity(enroll_emb, cohort_emb))
            
            if t_scores:
                t_mean = np.mean(t_scores)
                t_std = np.std(t_scores) + 1e-6
                t_normalized = (z_normalized - (t_mean - z_mean) / t_std)
        
        # Convert to probability
        normalized = 1.0 / (1.0 + np.exp(-t_normalized))
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _adaptive_snorm(self, raw_score: float, test_emb: np.ndarray) -> float:
        """Adaptive S-Norm: Top-N impostor scores."""
        if test_emb is None or len(self.cohort_embeddings) == 0:
            return self._cohort_norm(raw_score)
        
        cohort_scores = []
        for cohort_emb in self.cohort_embeddings:
            cohort_scores.append(self._raw_similarity(test_emb, cohort_emb))
        
        # Use top-N highest impostor scores
        cohort_scores_sorted = sorted(cohort_scores, reverse=True)
        top_scores = cohort_scores_sorted[:SNORM_TOP_N]
        
        if len(top_scores) > 0:
            s_mean = np.mean(top_scores)
            s_std = np.std(top_scores) + 1e-6
            normalized = (raw_score - s_mean) / s_std
            normalized = 1.0 / (1.0 + np.exp(-normalized * 2.0))
        else:
            normalized = raw_score
        
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _cohort_norm(self, raw_score: float) -> float:
        """Simple cohort normalization."""
        if self.impostor_mean is None:
            return raw_score
        
        normalized = (raw_score - self.impostor_mean) / self.impostor_std
        normalized = 1.0 / (1.0 + np.exp(-normalized * 2.0))
        
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _raw_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute raw cosine similarity."""
        emb1 = np.array(emb1, dtype=np.float64)
        emb2 = np.array(emb2, dtype=np.float64)
        
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        return float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
    
    def save(self, path: str = SCORE_NORM_MODEL_PATH) -> None:
        """Save score normalization model."""
        state = {
            'method': self.method,
            'is_trained': self.is_trained,
            'impostor_mean': self.impostor_mean,
            'impostor_std': self.impostor_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'cohort_embeddings': self.cohort_embeddings,
            'speaker_stats': self.speaker_stats,
            'num_speakers': self._num_speakers_at_train,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str = SCORE_NORM_MODEL_PATH) -> bool:
        """Load score normalization model."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.method = state['method']
            self.is_trained = state['is_trained']
            self.impostor_mean = state['impostor_mean']
            self.impostor_std = state['impostor_std']
            self.target_mean = state.get('target_mean')
            self.target_std = state.get('target_std')
            self.cohort_embeddings = state['cohort_embeddings']
            self.speaker_stats = state.get('speaker_stats', {})
            self._num_speakers_at_train = state.get('num_speakers', 0)
            print(f"   ✅ ScoreNorm loaded  |  method={self.method}  "
                  f"|  speakers_at_train={self._num_speakers_at_train}")
            return True
        except Exception as e:
            print(f"   ⚠️  ScoreNorm load error: {e}")
            return False


class ScoreNormManager:
    """Manager for score normalization training and application."""
    
    def __init__(self):
        self.normalizer = ScoreNormalizer(method=SCORE_NORM_METHOD)
        self._profiles_since_last_train = 0
        self._initialise()
    
    def _initialise(self):
        if not ENABLE_SCORE_NORMALIZATION:
            print("   ℹ️  ScoreNorm disabled via config.")
            return
        if self.normalizer.load(SCORE_NORM_MODEL_PATH):
            return
        self.retrain()
    
    def retrain(self):
        """Retrain score normalization from current database."""
        print("   🔄 ScoreNorm: collecting embeddings from DB…")
        speaker_embeddings = _collect_all_embeddings_from_db()
        
        if len(speaker_embeddings) < SCORE_NORM_MIN_SPEAKERS:
            print(f"   ℹ️  ScoreNorm: only {len(speaker_embeddings)} speaker(s) in DB "
                  f"(need ≥ {SCORE_NORM_MIN_SPEAKERS}). Skipping train.")
            self.normalizer.is_trained = False
            return
        
        print(f"   🔄 ScoreNorm: training on {len(speaker_embeddings)} speaker(s)…")
        self.normalizer.train(speaker_embeddings)
        self.normalizer.save(SCORE_NORM_MODEL_PATH)
        self._profiles_since_last_train = 0
    
    def on_new_profile(self):
        """Trigger retraining after N new profiles."""
        self._profiles_since_last_train += 1
        if self._profiles_since_last_train >= SCORE_NORM_RETRAIN_INTERVAL:
            self.retrain()
    
    @property
    def is_ready(self) -> bool:
        return ENABLE_SCORE_NORMALIZATION and self.normalizer.is_trained
    
    def normalize(self, raw_score: float, test_emb: np.ndarray = None,
                  enroll_emb: np.ndarray = None, speaker_id: int = None) -> float:
        """Apply score normalization if ready."""
        if not self.is_ready:
            return raw_score
        return self.normalizer.normalize(raw_score, test_emb, enroll_emb, speaker_id)


# =========================================================================
# WCCN (Within-Class Covariance Normalization)
# =========================================================================

class WCCN:
    """Within-Class Covariance Normalization for speaker recognition."""
    
    def __init__(self, embedding_dim: int = PLDA_EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.W: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.is_trained = False
        self._num_speakers_at_train = 0
    
    def train(self, speaker_embeddings: dict[int, list[np.ndarray]]) -> None:
        valid_speakers = {
            sid: embs for sid, embs in speaker_embeddings.items()
            if len(embs) >= WCCN_MIN_SAMPLES_PER_SPEAKER
        }
        
        if len(valid_speakers) < WCCN_MIN_SPEAKERS:
            print(f"   ℹ️  WCCN: need ≥ {WCCN_MIN_SPEAKERS} speakers with "
                  f"≥ {WCCN_MIN_SAMPLES_PER_SPEAKER} samples each. "
                  f"Current: {len(valid_speakers)} speakers")
            return
        
        all_embs = []
        for embs in valid_speakers.values():
            all_embs.extend(embs)
        all_embs = np.array(all_embs, dtype=np.float64)
        
        D = all_embs.shape[1]
        self.mean = all_embs.mean(axis=0)
        
        Sigma_w = np.zeros((D, D), dtype=np.float64)
        n_total = 0
        
        for sid, embs in valid_speakers.items():
            embs_arr = np.array(embs, dtype=np.float64)
            speaker_mean = embs_arr.mean(axis=0)
            centered = embs_arr - speaker_mean
            Sigma_w += centered.T @ centered
            n_total += len(embs)
        
        Sigma_w /= max(n_total, 1)
        Sigma_w += np.eye(D, dtype=np.float64) * WCCN_REGULARIZATION
        
        eigvals, eigvecs = np.linalg.eigh(Sigma_w)
        eigvals = np.maximum(eigvals, WCCN_REGULARIZATION)
        Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        self.W = eigvecs @ Lambda_inv_sqrt @ eigvecs.T
        
        self.is_trained = True
        self._num_speakers_at_train = len(valid_speakers)
        
        sample_counts = [len(e) for e in list(valid_speakers.values())[:5]]
        print(f"   ✅ WCCN trained  |  speakers={len(valid_speakers)}  "
              f"|  samples={sample_counts}...  |  reg={WCCN_REGULARIZATION}")
    
    def transform(self, embedding: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.W is None:
            return embedding
        
        embedding = embedding.astype(np.float64)
        centered = embedding - self.mean
        normalized = self.W @ centered
        norm = np.linalg.norm(normalized)
        if norm > 1e-8:
            normalized /= norm
        return normalized
    
    def save(self, path: str = WCCN_MODEL_PATH) -> None:
        state = {
            'embedding_dim': self.embedding_dim,
            'W': self.W,
            'mean': self.mean,
            'is_trained': self.is_trained,
            'num_speakers': self._num_speakers_at_train,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str = WCCN_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.embedding_dim = state['embedding_dim']
            self.W = state['W']
            self.mean = state['mean']
            self.is_trained = state['is_trained']
            self._num_speakers_at_train = state.get('num_speakers', 0)
            print(f"   ✅ WCCN model loaded  |  speakers_at_train={self._num_speakers_at_train}")
            return True
        except Exception as e:
            print(f"   ⚠️  WCCN load error: {e}")
            return False


class WCCNManager:
    def __init__(self):
        self.wccn = WCCN(embedding_dim=PLDA_EMBEDDING_DIM)
        self._profiles_since_last_train = 0
        self._initialise()
    
    def _initialise(self):
        if not ENABLE_WCCN:
            print("   ℹ️  WCCN disabled via config.")
            return
        if self.wccn.load(WCCN_MODEL_PATH):
            return
        self.retrain()
    
    def retrain(self):
        print("   🔄 WCCN: collecting embeddings from DB…")
        speaker_embeddings = _collect_all_embeddings_from_db()
        
        if len(speaker_embeddings) < WCCN_MIN_SPEAKERS:
            print(f"   ℹ️  WCCN: only {len(speaker_embeddings)} speaker(s) in DB "
                  f"(need ≥ {WCCN_MIN_SPEAKERS}). Skipping train.")
            self.wccn.is_trained = False
            return
        
        print(f"   🔄 WCCN: training on {len(speaker_embeddings)} speaker(s)…")
        self.wccn.train(speaker_embeddings)
        self.wccn.save(WCCN_MODEL_PATH)
        self._profiles_since_last_train = 0
    
    def on_new_profile(self):
        self._profiles_since_last_train += 1
        if self._profiles_since_last_train >= WCCN_RETRAIN_INTERVAL:
            self.retrain()
    
    @property
    def is_ready(self) -> bool:
        return ENABLE_WCCN and self.wccn.is_trained
    
    def transform(self, embedding: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return embedding
        return self.wccn.transform(embedding)


def _collect_all_embeddings_from_db() -> dict[int, list[np.ndarray]]:
    """Collect ALL embeddings from all centroids for training."""
    result: dict[int, list[np.ndarray]] = {}
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute('SELECT id, embedding FROM voice_profiles')
        rows = cursor.fetchall()
        conn.close()
        
        for pid, blob in rows:
            data = pickle.loads(blob)
            embeddings = []
            
            if isinstance(data, dict) and 'centroids' in data:
                from_dict_func = VoiceProfileCentroid.from_dict if 'VoiceProfileCentroid' in globals() else None
                if from_dict_func:
                    centroid_obj = from_dict_func(data)
                    for centroid in centroid_obj.centroids:
                        emb = centroid['resemblyzer_embedding']
                        embeddings.append(np.array(emb, dtype=np.float64))
            elif isinstance(data, dict):
                emb = data.get('resemblyzer_embedding', data.get('embedding'))
                if emb is not None:
                    embeddings.append(np.array(emb, dtype=np.float64))
            else:
                embeddings.append(np.array(data, dtype=np.float64))
            
            if embeddings:
                result[pid] = embeddings
                
    except Exception as e:
        print(f"   ⚠️  DB read error: {e}")
    return result


# =========================================================================
# MULTI-CENTROID CLUSTERING SYSTEM
# =========================================================================

class VoiceProfileCentroid:
    """Manages multiple centroids per profile for acoustic robustness."""
    
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
            self._update_centroid(best_idx, res_emb, ecapa_emb, quality, duration, embedding_data)
        elif len(self.centroids) < self.max_centroids:
            self._create_new_centroid(res_emb, ecapa_emb, quality, duration, embedding_data)
        else:
            self._update_centroid(best_idx, res_emb, ecapa_emb, quality, duration, embedding_data)
        
        if self.total_samples % 10 == 0:
            self._merge_similar_centroids()
            self._prune_weak_centroids()
        
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def _extract_resemblyzer(self, data):
        if isinstance(data, dict):
            return np.array(data.get('resemblyzer_embedding', 
                                    data.get('embedding', data)), dtype=np.float64)
        return np.array(data, dtype=np.float64)
    
    def _find_best_centroid(self, embedding):
        similarities = []
        for centroid in self.centroids:
            cos_sim = np.dot(embedding, centroid['resemblyzer_embedding'])
            euc_sim = 1.0 / (1.0 + np.linalg.norm(embedding - centroid['resemblyzer_embedding']))
            combined_sim = 0.7 * cos_sim + 0.3 * euc_sim
            similarities.append(combined_sim)
        
        best_idx = np.argmax(similarities)
        return best_idx, similarities[best_idx]
    
    def _create_new_centroid(self, res_emb, ecapa_emb, quality, duration, full_data):
        centroid = {
            'resemblyzer_embedding': res_emb.copy(),
            'ecapa_embedding': ecapa_emb.copy() if ecapa_emb is not None else None,
            'weight': quality * duration,
            'sample_count': 1,
            'quality_avg': quality,
            'duration_total': duration,
            'gender': full_data.get('gender', 'unknown') if isinstance(full_data, dict) else 'unknown',
            'created_at': datetime.now().isoformat(),
        }
        self.centroids.append(centroid)
    
    def _update_centroid(self, idx, res_emb, ecapa_emb, quality, duration, full_data):
        centroid = self.centroids[idx]
        
        alpha = min(0.15, max(0.02, 1.0 / (centroid['sample_count'] + 1)))
        alpha *= (0.7 + 0.3 * quality)
        
        old_res = centroid['resemblyzer_embedding']
        new_res = (1 - alpha) * old_res + alpha * res_emb
        new_res /= (np.linalg.norm(new_res) + 1e-8)
        centroid['resemblyzer_embedding'] = new_res
        
        if ecapa_emb is not None and centroid['ecapa_embedding'] is not None:
            old_ecapa = centroid['ecapa_embedding']
            new_ecapa = (1 - alpha) * old_ecapa + alpha * ecapa_emb
            new_ecapa /= (np.linalg.norm(new_ecapa) + 1e-8)
            centroid['ecapa_embedding'] = new_ecapa
        elif ecapa_emb is not None:
            centroid['ecapa_embedding'] = ecapa_emb.copy()
        
        centroid['sample_count'] += 1
        centroid['weight'] += quality * duration
        centroid['quality_avg'] = (centroid['quality_avg'] * (centroid['sample_count'] - 1) + quality) / centroid['sample_count']
        centroid['duration_total'] += duration
    
    def _merge_similar_centroids(self):
        if len(self.centroids) <= 1:
            return
        
        i = 0
        while i < len(self.centroids) - 1:
            j = i + 1
            while j < len(self.centroids):
                emb_i = self.centroids[i]['resemblyzer_embedding']
                emb_j = self.centroids[j]['resemblyzer_embedding']
                similarity = np.dot(emb_i, emb_j)
                
                if similarity >= CENTROID_MERGE_THRESHOLD:
                    count_i = self.centroids[i]['sample_count']
                    count_j = self.centroids[j]['sample_count']
                    total = count_i + count_j
                    
                    weight_i = count_i / total
                    weight_j = count_j / total
                    
                    merged_res = weight_i * emb_i + weight_j * emb_j
                    merged_res /= (np.linalg.norm(merged_res) + 1e-8)
                    self.centroids[i]['resemblyzer_embedding'] = merged_res
                    
                    if (self.centroids[i]['ecapa_embedding'] is not None and 
                        self.centroids[j]['ecapa_embedding'] is not None):
                        ecapa_i = self.centroids[i]['ecapa_embedding']
                        ecapa_j = self.centroids[j]['ecapa_embedding']
                        merged_ecapa = weight_i * ecapa_i + weight_j * ecapa_j
                        merged_ecapa /= (np.linalg.norm(merged_ecapa) + 1e-8)
                        self.centroids[i]['ecapa_embedding'] = merged_ecapa
                    
                    self.centroids[i]['sample_count'] = total
                    self.centroids[i]['weight'] += self.centroids[j]['weight']
                    self.centroids[i]['quality_avg'] = (
                        weight_i * self.centroids[i]['quality_avg'] + 
                        weight_j * self.centroids[j]['quality_avg']
                    )
                    self.centroids[i]['duration_total'] += self.centroids[j]['duration_total']
                    
                    del self.centroids[j]
                else:
                    j += 1
            i += 1
    
    def _prune_weak_centroids(self):
        if len(self.centroids) <= 2:
            return
        
        counts = [c['sample_count'] for c in self.centroids]
        median_count = np.median(counts)
        
        self.centroids = [
            c for c in self.centroids 
            if c['sample_count'] >= max(CENTROID_MIN_SAMPLES, median_count * 0.15)
        ]
    
    def get_best_match_score(self, embedding_data, use_plda=False, plda_manager=None, 
                            wccn_manager=None, score_norm_manager=None, speaker_id=None):
        """Calculate similarity with WCCN, PLDA, and score normalization."""
        if not self.centroids:
            return 0.0
        
        res_emb = self._extract_resemblyzer(embedding_data)
        
        # Apply WCCN transformation
        if wccn_manager and wccn_manager.is_ready:
            res_emb = wccn_manager.transform(res_emb)
        
        ecapa_emb = embedding_data.get('ecapa_embedding') if isinstance(embedding_data, dict) else None
        
        scores = []
        weights = []
        
        for centroid in self.centroids:
            cent_res = centroid['resemblyzer_embedding']
            
            # Apply WCCN to stored centroid
            if wccn_manager and wccn_manager.is_ready:
                cent_res = wccn_manager.transform(cent_res)
            
            cos_sim = float(np.clip(np.dot(res_emb, cent_res), -1.0, 1.0))
            euc_sim = 1.0 / (1.0 + np.linalg.norm(res_emb - cent_res))
            pearson_sim = float(np.clip(np.corrcoef(res_emb, cent_res)[0, 1], -1.0, 1.0))
            
            resemblyzer_score = 0.65 * cos_sim + 0.25 * euc_sim + 0.10 * pearson_sim
            
            ecapa_score = None
            if ENABLE_ECAPA and ecapa_emb is not None and centroid['ecapa_embedding'] is not None:
                ecapa_score = float(np.clip(np.dot(ecapa_emb, centroid['ecapa_embedding']), -1.0, 1.0))
            
            if ecapa_score is not None:
                base_score = ECAPA_WEIGHT * ecapa_score + (1 - ECAPA_WEIGHT) * resemblyzer_score
            else:
                base_score = resemblyzer_score
            
            if use_plda and plda_manager is not None and plda_manager.is_ready:
                plda_prob = plda_manager.score_prob(res_emb, cent_res)
                final_score = PLDA_SCORE_WEIGHT * plda_prob + (1 - PLDA_SCORE_WEIGHT) * base_score
            else:
                final_score = base_score
            
            # Apply score normalization
            if score_norm_manager and score_norm_manager.is_ready:
                final_score = score_norm_manager.normalize(
                    final_score, 
                    test_emb=res_emb, 
                    enroll_emb=cent_res,
                    speaker_id=speaker_id
                )
            
            scores.append(final_score)
            
            centroid_weight = (
                centroid['quality_avg'] ** CENTROID_QUALITY_WEIGHT * 
                np.log1p(centroid['sample_count'])
            )
            weights.append(centroid_weight)
        
        scores = np.array(scores)
        weights = np.array(weights)
        weights /= weights.sum()
        
        best_score = np.max(scores)
        weighted_avg = np.average(scores, weights=weights)
        
        final_score = 0.85 * best_score + 0.15 * weighted_avg
        
        return float(final_score)
    
    def get_primary_centroid(self):
        if not self.centroids:
            return None
        return max(self.centroids, key=lambda c: c['sample_count'])
    
    def get_stats(self):
        return {
            'num_centroids': len(self.centroids),
            'total_samples': self.total_samples,
            'sample_counts': [c['sample_count'] for c in self.centroids],
            'quality_averages': [c['quality_avg'] for c in self.centroids],
        }
    
    def to_dict(self):
        return {
            'centroids': self.centroids,
            'total_samples': self.total_samples,
            'metadata': self.metadata,
        }
    
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
        
        self.match_history = []
        self.threshold_adjustments = []
        self.profile_thresholds = {}
        
        self._load_state()
    
    def _load_state(self):
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return
        
        if os.path.exists(THRESHOLD_STATE_PATH):
            try:
                with open(THRESHOLD_STATE_PATH, 'rb') as f:
                    state = pickle.load(f)
                self.similarity_threshold = state.get('similarity_threshold', BASE_SIMILARITY_THRESHOLD)
                self.strong_match_threshold = state.get('strong_match_threshold', BASE_STRONG_MATCH_THRESHOLD)
                self.verification_threshold = state.get('verification_threshold', BASE_VERIFICATION_THRESHOLD)
                self.profile_thresholds = state.get('profile_thresholds', {})
                print(f"   ✅ Adaptive thresholds loaded  |  base={self.similarity_threshold:.3f}")
            except Exception as e:
                print(f"   ⚠️  Threshold state load error: {e}")
    
    def _save_state(self):
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return
        
        try:
            state = {
                'similarity_threshold': self.similarity_threshold,
                'strong_match_threshold': self.strong_match_threshold,
                'verification_threshold': self.verification_threshold,
                'profile_thresholds': self.profile_thresholds,
                'last_updated': datetime.now().isoformat(),
            }
            with open(THRESHOLD_STATE_PATH, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"   ⚠️  Threshold state save error: {e}")
    
    def get_threshold_for_profile(self, profile_id: int, total_recordings: int, 
                                  num_centroids: int = 1) -> float:
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return self.similarity_threshold
        
        base = self.similarity_threshold
        
        if profile_id in self.profile_thresholds:
            base = self.profile_thresholds[profile_id]
        
        if total_recordings >= PROFILE_MATURITY_THRESHOLD:
            maturity_factor = min((total_recordings - PROFILE_MATURITY_THRESHOLD) / 20.0, 0.10)
            base += maturity_factor
        
        if num_centroids > 2:
            diversity_factor = min((num_centroids - 2) * 0.015, 0.05)
            base += diversity_factor
        
        return float(np.clip(base, THRESHOLD_MIN, THRESHOLD_MAX))
    
    def analyze_and_adjust(self, all_similarities: list[tuple[float, int, str]], 
                          best_match_id: int | None) -> None:
        if not ENABLE_ADAPTIVE_THRESHOLDS or len(all_similarities) < 2:
            return
        
        scores = [s[0] for s in all_similarities]
        scores_sorted = sorted(scores, reverse=True)
        
        if len(scores_sorted) >= 2:
            best = scores_sorted[0]
            second = scores_sorted[1]
            gap = best - second
            
            if gap > SIMILARITY_GAP_THRESHOLD and best > self.similarity_threshold + 0.05:
                self._adjust_threshold(THRESHOLD_STEP, f"High confidence (gap={gap:.3f})")
            elif gap < 0.05 and best < self.similarity_threshold + 0.03:
                self._adjust_threshold(-THRESHOLD_STEP, f"Low confidence (gap={gap:.3f})")
        
        num_profiles = self._get_database_size()
        if num_profiles > 20:
            target_threshold = BASE_SIMILARITY_THRESHOLD + min((num_profiles - 20) / 100.0, 0.08)
            if self.similarity_threshold < target_threshold - 0.01:
                self._adjust_threshold(THRESHOLD_STEP, f"DB scaling ({num_profiles} profiles)")
    
    def _adjust_threshold(self, delta: float, reason: str) -> None:
        old = self.similarity_threshold
        self.similarity_threshold = float(np.clip(old + delta, THRESHOLD_MIN, THRESHOLD_MAX))
        
        if abs(self.similarity_threshold - old) > 1e-6:
            self.threshold_adjustments.append((datetime.now().isoformat(), reason, delta))
            if len(self.threshold_adjustments) > 100:
                self.threshold_adjustments = self.threshold_adjustments[-100:]
            
            ratio = self.similarity_threshold / old if old > 0 else 1.0
            self.strong_match_threshold *= ratio
            self.strong_match_threshold = float(np.clip(
                self.strong_match_threshold, THRESHOLD_MIN + 0.05, THRESHOLD_MAX
            ))
            self.verification_threshold *= ratio
            self.verification_threshold = float(np.clip(
                self.verification_threshold, THRESHOLD_MIN + 0.02, THRESHOLD_MAX - 0.05
            ))
            
            self._save_state()
            print(f"   🎯 Threshold adjusted: {old:.3f} → {self.similarity_threshold:.3f} ({reason})")
    
    def _get_database_size(self) -> int:
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            count = conn.execute('SELECT COUNT(*) FROM voice_profiles').fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0
    
    def on_match_confirmed(self, profile_id: int, similarity: float, was_correct: bool) -> None:
        if not ENABLE_ADAPTIVE_THRESHOLDS:
            return
        
        current_threshold = self.profile_thresholds.get(profile_id, self.similarity_threshold)
        
        if was_correct and similarity < current_threshold + 0.05:
            new_threshold = current_threshold - abs(FALSE_REJECT_REWARD)
        elif not was_correct and similarity > current_threshold:
            new_threshold = current_threshold + FALSE_ACCEPT_PENALTY
        else:
            return
        
        new_threshold = float(np.clip(new_threshold, THRESHOLD_MIN, THRESHOLD_MAX))
        self.profile_thresholds[profile_id] = new_threshold
        self._save_state()
    
    def get_stats(self) -> dict:
        return {
            'base_threshold': self.similarity_threshold,
            'strong_match': self.strong_match_threshold,
            'verification': self.verification_threshold,
            'profile_overrides': len(self.profile_thresholds),
            'recent_adjustments': len(self.threshold_adjustments),
        }


threshold_manager = AdaptiveThresholdManager()


# =========================================================================
# PLDA
# =========================================================================

class PLDA:
    """Probabilistic Linear Discriminant Analysis for speaker scoring."""
    
    def __init__(self, embedding_dim: int = PLDA_EMBEDDING_DIM,
                 latent_dim: int = PLDA_LATENT_DIM):
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.mean: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.Sigma_w: np.ndarray | None = None
        self.is_trained = False
        self._num_speakers_at_train = 0
    
    def train(self, speaker_embeddings: dict[int, list[np.ndarray]], n_iter: int = 10) -> None:
        all_embs = []
        for embs in speaker_embeddings.values():
            all_embs.extend(embs)
        all_embs = np.array(all_embs, dtype=np.float64)
        
        if all_embs.shape[0] < 2:
            print("   ⚠️  PLDA: need ≥ 2 total embeddings to train.")
            return
        
        D = all_embs.shape[1]
        self.mean = all_embs.mean(axis=0)
        
        Sw = np.zeros((D, D), dtype=np.float64)
        n_total = 0
        speaker_means = {}
        
        for sid, embs in speaker_embeddings.items():
            embs_arr = np.array(embs, dtype=np.float64)
            sp_mean = embs_arr.mean(axis=0)
            speaker_means[sid] = sp_mean
            diff = embs_arr - sp_mean
            Sw += diff.T @ diff
            n_total += len(embs)
        
        Sw /= max(n_total, 1)
        Sw += np.eye(D, dtype=np.float64) * 1e-4
        
        sp_mean_mat = np.array(list(speaker_means.values()), dtype=np.float64)
        sp_mean_centred = sp_mean_mat - self.mean
        Sb = (sp_mean_centred.T @ sp_mean_centred) / max(len(speaker_means), 1)
        
        eigvals, eigvecs = np.linalg.eigh(Sb)
        k = min(self.latent_dim, D)
        idx = np.argsort(eigvals)[::-1][:k]
        V = eigvecs[:, idx].T.copy()
        
        Sw_inv = np.linalg.inv(Sw)
        
        for iteration in range(n_iter):
            VSwi = V @ Sw_inv
            A = np.eye(k, dtype=np.float64) + VSwi @ V.T
            A_inv = np.linalg.inv(A)
            
            E_z = {}
            E_zzT = {}
            
            for sid, embs in speaker_embeddings.items():
                embs_arr = np.array(embs, dtype=np.float64)
                m_s = embs_arr.mean(axis=0) - self.mean
                ez = A_inv @ (VSwi @ m_s)
                E_z[sid] = ez
                n_i = len(embs)
                E_zzT[sid] = A_inv + np.outer(ez, ez) * n_i
            
            num_V = np.zeros((k, D), dtype=np.float64)
            den_V = np.zeros((k, k), dtype=np.float64)
            Sw_new = np.zeros((D, D), dtype=np.float64)
            n_total = 0
            
            for sid, embs in speaker_embeddings.items():
                embs_arr = np.array(embs, dtype=np.float64)
                n_i = len(embs)
                m_s = embs_arr.mean(axis=0) - self.mean
                ez = E_z[sid]
                ezzT = E_zzT[sid]
                
                num_V += n_i * np.outer(ez, m_s)
                den_V += ezzT
                
                diff = embs_arr - (self.mean + V.T @ ez)
                Sw_new += diff.T @ diff
                n_total += n_i
            
            den_V_inv = np.linalg.inv(den_V + np.eye(k) * 1e-6)
            V = den_V_inv @ num_V
            
            Sw = Sw_new / max(n_total, 1)
            Sw += np.eye(D, dtype=np.float64) * 1e-4
            Sw_inv = np.linalg.inv(Sw)
        
        self.V = V
        self.Sigma_w = Sw
        self.is_trained = True
        self._num_speakers_at_train = len(speaker_embeddings)
        self._llr_scale = self._calibrate_scale(speaker_embeddings)
        
        print(f"   ✅ PLDA trained  |  speakers={len(speaker_embeddings)}  "
              f"|  latent_dim={k}  |  iters={n_iter}  |  llr_scale={self._llr_scale:.2f}")
    
    def _calibrate_scale(self, speaker_embeddings: dict[int, list[np.ndarray]], 
                         n_samples: int = 30) -> float:
        sids = list(speaker_embeddings.keys())
        if len(sids) < 2:
            return 1.0
        
        rng = np.random.default_rng(0)
        llrs = []
        for _ in range(n_samples):
            s = rng.choice(sids)
            d = s
            while d == s:
                d = rng.choice(sids)
            embs_s = speaker_embeddings[s]
            i, j = rng.choice(len(embs_s), 2, replace=(len(embs_s) < 2))
            llrs.append(abs(self._raw_score(embs_s[i], embs_s[j])))
            embs_d = speaker_embeddings[d]
            llrs.append(abs(self._raw_score(embs_s[0], embs_d[0])))
        
        scale = float(np.median(llrs))
        return max(scale, 1e-6)
    
    def score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        raw = self._raw_score(emb1, emb2)
        scale = getattr(self, '_llr_scale', 1.0)
        return raw / scale
    
    def _raw_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if not self.is_trained or self.V is None or self.Sigma_w is None:
            return 0.0
        
        emb1 = emb1.astype(np.float64)
        emb2 = emb2.astype(np.float64)
        
        D = self.embedding_dim
        
        x1 = emb1 - self.mean
        x2 = emb2 - self.mean
        
        Sw_inv = np.linalg.inv(self.Sigma_w)
        VtV = self.V.T @ self.V
        
        Sigma_s = self.Sigma_w + VtV
        Sigma_s_inv = np.linalg.inv(Sigma_s + np.eye(D) * 1e-6)
        
        _sign_w, logdet_w = np.linalg.slogdet(self.Sigma_w)
        _sign_s, logdet_s = np.linalg.slogdet(Sigma_s)
        
        quad_d = (x1 @ Sw_inv @ x1 + x2 @ Sw_inv @ x2) * 0.5
        
        x_sum = x1 + x2
        x_diff = x1 - x2
        quad_s = (x_diff @ Sw_inv @ x_diff + x_sum @ Sigma_s_inv @ x_sum) * 0.25
        
        log_norm_d = logdet_w
        log_norm_s = (logdet_w + logdet_s) * 0.5
        
        llr = (log_norm_d - log_norm_s) + (quad_d - quad_s)
        return float(llr)
    
    @staticmethod
    def sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            ez = np.exp(x)
            return ez / (1.0 + ez)
    
    def score_normalised(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return self.sigmoid(self.score(emb1, emb2))
    
    def save(self, path: str = PLDA_MODEL_PATH) -> None:
        state = {
            'embedding_dim': self.embedding_dim,
            'latent_dim': self.latent_dim,
            'mean': self.mean,
            'V': self.V,
            'Sigma_w': self.Sigma_w,
            'is_trained': self.is_trained,
            'num_speakers': self._num_speakers_at_train,
            'llr_scale': getattr(self, '_llr_scale', 1.0),
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str = PLDA_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.embedding_dim = state['embedding_dim']
            self.latent_dim = state['latent_dim']
            self.mean = state['mean']
            self.V = state['V']
            self.Sigma_w = state['Sigma_w']
            self.is_trained = state['is_trained']
            self._num_speakers_at_train = state.get('num_speakers', 0)
            self._llr_scale = state.get('llr_scale', 1.0)
            print(f"   ✅ PLDA model loaded  |  speakers_at_train={self._num_speakers_at_train}")
            return True
        except Exception as e:
            print(f"   ⚠️  PLDA load error: {e}")
            return False


class PLDAManager:
    def __init__(self):
        self.plda = PLDA(embedding_dim=PLDA_EMBEDDING_DIM, latent_dim=PLDA_LATENT_DIM)
        self._profiles_since_last_train = 0
        self._initialise()
    
    def _initialise(self):
        if not ENABLE_PLDA:
            print("   ℹ️  PLDA disabled via config.")
            return
        if self.plda.load(PLDA_MODEL_PATH):
            return
        self.retrain()
    
    def retrain(self):
        print("   🔄 PLDA: collecting embeddings from DB…")
        speaker_embeddings = _collect_all_embeddings_from_db()
        
        if len(speaker_embeddings) < PLDA_MIN_SPEAKERS:
            print(f"   ℹ️  PLDA: only {len(speaker_embeddings)} speaker(s) in DB "
                  f"(need ≥ {PLDA_MIN_SPEAKERS}). Skipping train.")
            self.plda.is_trained = False
            return
        
        print(f"   🔄 PLDA: training on {len(speaker_embeddings)} speaker(s)…")
        self.plda.train(speaker_embeddings)
        self.plda.save(PLDA_MODEL_PATH)
        self._profiles_since_last_train = 0
    
    def on_new_profile(self):
        self._profiles_since_last_train += 1
        if self._profiles_since_last_train >= PLDA_RETRAIN_INTERVAL:
            self.retrain()
    
    @property
    def is_ready(self) -> bool:
        return ENABLE_PLDA and self.plda.is_trained
    
    def score_llr(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if not self.is_ready:
            return 0.0
        return self.plda.score(emb1, emb2)
    
    def score_prob(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if not self.is_ready:
            return 0.5
        return self.plda.score_normalised(emb1, emb2)


# =========================================================================
# INITIALIZATION
# =========================================================================

print("🚀 Loading Ultra-Accurate Voice Recognition System (Score Norm Edition) …")

try:
    encoder = VoiceEncoder()
    print("✅ Resemblyzer encoder loaded")
except Exception as e:
    print(f"❌ Resemblyzer error: {e}")
    encoder = None

ecapa_classifier = None
if ENABLE_ECAPA and EncoderClassifier is not None:
    try:
        print("📥 Loading ECAPA-TDNN model …")
        ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        print("✅ ECAPA-TDNN loaded")
    except Exception as e:
        print(f"   ⚠️  ECAPA-TDNN failed ({e})  →  running without it")
        ecapa_classifier = None
        ENABLE_ECAPA = False
else:
    ENABLE_ECAPA = False

diarization_pipeline = None
try:
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("⚠️ No HUGGINGFACE_TOKEN in .env")
    else:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token
        )
        diarization_pipeline.to(torch.device("cpu"))
        num_cores = multiprocessing.cpu_count()
        torch.set_num_threads(num_cores)
        torch.set_grad_enabled(False)
        print(f"✅ Pyannote pipeline loaded ({num_cores} cores)")
except Exception as e:
    print(f"❌ Pyannote error: {e}")

vad = None
if ENABLE_VAD:
    try:
        vad = webrtcvad.Vad(3)
        print("✅ VAD enabled (aggressiveness 3)")
    except Exception:
        ENABLE_VAD = False
        print("⚠️ VAD not available")

print("📥 Initialising PLDA scorer …")
plda_manager = PLDAManager()

print("📥 Initialising WCCN normalizer …")
wccn_manager = WCCNManager()

print("📥 Initialising Score Normalizer …")
score_norm_manager = ScoreNormManager()

print(f"\n{'='*70}")
print("🎯 CONFIGURATION:")
print(f"   • Resemblyzer          : ✅")
print(f"   • ECAPA-TDNN           : {'✅' if ENABLE_ECAPA else '❌ (fallback mode)'}")
print(f"   • PLDA scorer          : {'✅  trained' if plda_manager.is_ready else '⏳  waiting for data'}")
print(f"   • WCCN normalizer      : {'✅  trained' if wccn_manager.is_ready else '⏳  waiting for data'}")
print(f"   • Score Normalizer     : {'✅  (' + SCORE_NORM_METHOD + ')' if score_norm_manager.is_ready else '⏳  waiting for data'}")
print(f"   • Multi-Centroid System: {'✅  (max ' + str(MAX_CENTROIDS_PER_PROFILE) + ' per profile)' if ENABLE_MULTI_CENTROID else '❌'}")
print(f"   • Adaptive Thresholds  : {'✅' if ENABLE_ADAPTIVE_THRESHOLDS else '❌'}")
if ENABLE_ADAPTIVE_THRESHOLDS:
    print(f"   • Current Threshold    : {threshold_manager.similarity_threshold:.3f}")
print(f"   • VAD                  : {'✅' if ENABLE_VAD else '❌'}")
print(f"   • Gender Classification: {'✅' if ENABLE_GENDER_CLASSIFICATION else '❌'}")
print(f"   • Quality Scoring      : {'✅' if ENABLE_QUALITY_SCORING else '❌'}")
print(f"{'='*70}\n")


# =========================================================================
# AUDIO PROCESSING FUNCTIONS
# =========================================================================

def apply_vad(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_VAD or vad is None:
        return wav
    try:
        wav_int16 = (wav * 32767).astype(np.int16)
        frame_duration = 30
        frame_length = int(sample_rate * frame_duration / 1000)
        speech_frames = []
        for i in range(0, len(wav_int16) - frame_length, frame_length):
            frame = wav_int16[i:i + frame_length].tobytes()
            if vad.is_speech(frame, sample_rate):
                speech_frames.append(wav[i:i + frame_length])
        return np.concatenate(speech_frames) if speech_frames else wav
    except Exception as e:
        print(f"⚠️ VAD error: {e}")
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
        print(f"⚠️ Filtering error: {e}")
        wav = wav - np.mean(wav)
        mx = np.max(np.abs(wav))
        return wav / mx if mx > 0 else wav


def calculate_audio_quality(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_QUALITY_SCORING or len(wav) < sample_rate:
        return 1.0
    try:
        noise_n = min(int(0.2 * sample_rate), len(wav) // 4)
        noise_power = np.mean(wav[:noise_n] ** 2) + 1e-10
        sig_power = np.mean(wav[noise_n:] ** 2) + 1e-10
        snr = 10 * np.log10(sig_power / noise_power)
        snr_score = float(np.clip((snr - 5) / 20, 0, 1))
        dr_score = float(np.clip(np.std(wav) / 0.15, 0, 1))
        clip_score = 1.0 - float(np.clip(np.sum(np.abs(wav) > 0.95) / len(wav) * 50, 0, 1))
        zcr = np.sum(np.abs(np.diff(np.sign(wav)))) / len(wav)
        zcr_score = float(np.clip(zcr / 0.1, 0, 1))
        return 0.40 * snr_score + 0.25 * dr_score + 0.25 * clip_score + 0.10 * zcr_score
    except Exception:
        return 0.5


def classify_gender(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_GENDER_CLASSIFICATION or len(wav) < sample_rate:
        return 'unknown'
    try:
        f0 = librosa.yin(wav, fmin=75, fmax=400, sr=sample_rate)
        f0_voiced = f0[f0 > 75]
        if len(f0_voiced) < 10:
            return 'unknown'
        med = np.median(f0_voiced)
        if med < 145:
            return 'male'
        elif med > 165:
            return 'female'
        return 'unknown'
    except Exception:
        return 'unknown'


def extract_ecapa_embedding(wav, sample_rate=SAMPLE_RATE):
    if not ENABLE_ECAPA or ecapa_classifier is None:
        return None
    try:
        if sample_rate != 16000:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        with torch.no_grad():
            emb = ecapa_classifier.encode_batch(wav_tensor).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)
    except Exception as e:
        print(f"⚠️ ECAPA extraction error: {e}")
        return None


def extract_voice_embedding(audio_path, start_time=None, end_time=None):
    if encoder is None:
        return None
    try:
        wav = preprocess_wav(Path(audio_path))
        
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
        print(f"⚠️ Embedding extraction error: {e}")
        return None


def perform_smart_diarization(audio_path):
    if diarization_pipeline is None:
        return None
    try:
        print("🔍 Analyzing audio for speakers…")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(y) / sr
        print(f"📊 Audio duration: {duration:.1f}s ({duration/60:.1f} min)")
        
        temp_wav = audio_path.replace(os.path.splitext(audio_path)[1], '_temp.wav')
        sf.write(temp_wav, y, SAMPLE_RATE, subtype='PCM_16')
        
        with torch.no_grad():
            print("   Strategy 1: Unconstrained detection…")
            diarization_output = diarization_pipeline(temp_wav)
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        speakers_data = {}
        for segment, _, label in diarization_output.speaker_diarization.itertracks(yield_label=True):
            speakers_data.setdefault(label, []).append({
                'start': segment.start,
                'end': segment.end,
                'duration': segment.end - segment.start,
            })
        
        print(f"   ✅ Found {len(speakers_data)} speaker(s)")
        
        needs_retry = (
            (len(speakers_data) == 1 and duration > 15) or
            (duration > 180 and len(speakers_data) < 3) or
            (duration > 3600)
        )
        if needs_retry:
            min_spk = 2
            print(f"   Strategy 2: Retrying with min_speakers={min_spk}…")
            sf.write(temp_wav, y, SAMPLE_RATE, subtype='PCM_16')
            with torch.no_grad():
                diarization_output = diarization_pipeline(
                    temp_wav, min_speakers=min_spk, max_speakers=20
                )
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            
            speakers_data = {}
            for segment, _, label in diarization_output.speaker_diarization.itertracks(yield_label=True):
                speakers_data.setdefault(label, []).append({
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.end - segment.start,
                })
            print(f"   ✅ Retry found {len(speakers_data)} speaker(s)")
        
        filtered = {}
        for label, segs in speakers_data.items():
            total = sum(s['duration'] for s in segs)
            if total >= MIN_SPEAKING_TIME:
                filtered[label] = {'segments': segs, 'total_duration': total}
            else:
                print(f"   ⚠️ Filtered: {label} ({total:.2f}s < {MIN_SPEAKING_TIME}s)")
        
        print(f"✅ Final: {len(filtered)} speaker(s) detected")
        return filtered or None
    
    except Exception as e:
        print(f"❌ Diarization error: {e}")
        import traceback; traceback.print_exc()
        temp_wav = audio_path.replace(os.path.splitext(audio_path)[1], '_temp.wav')
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return None


# =========================================================================
# SIMILARITY & MATCHING with SCORE NORMALIZATION
# =========================================================================

def find_matching_speaker(embedding_data):
    """Find matching speaker using all enhancements."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        profiles = conn.execute('SELECT id, name, embedding, total_recordings FROM voice_profiles').fetchall()
        conn.close()
        if not profiles:
            return None
        
        all_similarities = []
        best_match, best_sim = None, 0.0
        
        for pid, name, blob, total_recordings in profiles:
            stored = pickle.loads(blob)
            
            if isinstance(stored, dict) and 'centroids' in stored:
                centroid_obj = VoiceProfileCentroid.from_dict(stored)
                sim = centroid_obj.get_best_match_score(
                    embedding_data, 
                    use_plda=True, 
                    plda_manager=plda_manager,
                    wccn_manager=wccn_manager,
                    score_norm_manager=score_norm_manager,
                    speaker_id=pid
                )
                num_centroids = len(centroid_obj.centroids)
            else:
                sim = calculate_similarity_legacy(embedding_data, stored)
                num_centroids = 1
            
            all_similarities.append((sim, pid, name))
            
            threshold = threshold_manager.get_threshold_for_profile(
                pid, total_recordings, num_centroids
            )
            
            if sim > threshold and sim > best_sim:
                verified, vsim = verify_speaker_match(embedding_data, pid)
                if verified:
                    best_sim = sim
                    best_match = {
                        'id': pid, 'name': name,
                        'confidence': sim * 100,
                        'similarity': sim, 'verified': True,
                        'num_centroids': num_centroids,
                    }
        
        threshold_manager.analyze_and_adjust(all_similarities, 
                                             best_match['id'] if best_match else None)
        
        return best_match
    except Exception as e:
        print(f"Error finding match: {e}")
        import traceback; traceback.print_exc()
        return None


def calculate_similarity_legacy(emb1_data, emb2_data) -> float:
    """Legacy similarity calculation."""
    r1 = _extract_resemblyzer(emb1_data)
    r2 = _extract_resemblyzer(emb2_data)
    
    # Apply WCCN if available
    if wccn_manager.is_ready:
        r1 = wccn_manager.transform(r1)
        r2 = wccn_manager.transform(r2)
    
    cosine = float(np.clip(np.dot(r1, r2), -1.0, 1.0))
    euclid = 1.0 / (1.0 + np.linalg.norm(r1 - r2))
    pearson = float(np.clip(np.corrcoef(r1, r2)[0, 1], -1.0, 1.0)) if len(r1) > 1 else cosine
    
    resemblyzer_score = 0.65 * cosine + 0.25 * euclid + 0.10 * pearson
    
    ecapa_score = None
    if ENABLE_ECAPA and isinstance(emb1_data, dict) and isinstance(emb2_data, dict):
        e1 = emb1_data.get('ecapa_embedding')
        e2 = emb2_data.get('ecapa_embedding')
        if e1 is not None and e2 is not None:
            ecapa_score = float(np.clip(np.dot(e1, e2), -1.0, 1.0))
    
    if ecapa_score is not None:
        base_score = ECAPA_WEIGHT * ecapa_score + (1 - ECAPA_WEIGHT) * resemblyzer_score
    else:
        base_score = resemblyzer_score
    
    if plda_manager.is_ready:
        plda_prob = plda_manager.score_prob(r1, r2)
        final_score = PLDA_SCORE_WEIGHT * plda_prob + (1 - PLDA_SCORE_WEIGHT) * base_score
    else:
        final_score = base_score
    
    # Apply score normalization
    if score_norm_manager.is_ready:
        final_score = score_norm_manager.normalize(final_score, test_emb=r1)
    
    if isinstance(emb1_data, dict) and isinstance(emb2_data, dict):
        g1 = emb1_data.get('gender', 'unknown')
        g2 = emb2_data.get('gender', 'unknown')
        if g1 != 'unknown' and g2 != 'unknown' and g1 != g2:
            final_score *= 0.85
    
    return float(final_score)


def _extract_resemblyzer(data):
    if isinstance(data, dict):
        return np.array(data.get('resemblyzer_embedding',
                                 data.get('embedding', data)), dtype=np.float64)
    return np.array(data, dtype=np.float64)


def verify_speaker_match(embedding_data, profile_id) -> tuple[bool, float]:
    """Verify speaker match with all enhancements."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        row = conn.execute('SELECT embedding, total_recordings FROM voice_profiles WHERE id=?',
                           (profile_id,)).fetchone()
        conn.close()
        if not row:
            return False, 0.0
        
        stored = pickle.loads(row[0])
        total_recordings = row[1]
        
        if isinstance(stored, dict) and 'centroids' in stored:
            centroid_obj = VoiceProfileCentroid.from_dict(stored)
            similarity = centroid_obj.get_best_match_score(
                embedding_data, 
                use_plda=True, 
                plda_manager=plda_manager,
                wccn_manager=wccn_manager,
                score_norm_manager=score_norm_manager,
                speaker_id=profile_id
            )
            num_centroids = len(centroid_obj.centroids)
        else:
            similarity = calculate_similarity_legacy(embedding_data, stored)
            num_centroids = 1
        
        threshold = threshold_manager.get_threshold_for_profile(
            profile_id, total_recordings, num_centroids
        )
        
        r_new = _extract_resemblyzer(embedding_data)
        primary_centroid = centroid_obj.get_primary_centroid() if isinstance(stored, dict) and 'centroids' in stored else stored
        r_old = _extract_resemblyzer(primary_centroid)
        
        # Apply WCCN
        if wccn_manager.is_ready:
            r_new = wccn_manager.transform(r_new)
            r_old = wccn_manager.transform(r_old)
        
        cos_ok = np.dot(r_new, r_old) >= 0.70
        euc_ok = (1.0 / (1.0 + np.linalg.norm(r_new - r_old))) >= 0.65
        
        plda_ok = True
        if plda_manager.is_ready:
            llr = plda_manager.score_llr(r_new, r_old)
            plda_ok = llr >= 0.0
        
        verified = (similarity >= threshold and cos_ok and euc_ok and plda_ok)
        return verified, similarity
    except Exception as e:
        print(f"⚠️ Verification error: {e}")
        return False, 0.0


# =========================================================================
# DATABASE OPERATIONS
# =========================================================================

def create_voice_profile(name, embedding_data):
    """Create new profile with multi-centroid system."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        now = datetime.now().isoformat()
        
        if ENABLE_MULTI_CENTROID:
            centroid_obj = VoiceProfileCentroid(max_centroids=MAX_CENTROIDS_PER_PROFILE)
            quality = embedding_data.get('quality', 1.0) if isinstance(embedding_data, dict) else 1.0
            duration = embedding_data.get('duration', 1.0) if isinstance(embedding_data, dict) else 1.0
            centroid_obj.add_embedding(embedding_data, quality, duration)
            stored_data = centroid_obj.to_dict()
        else:
            stored_data = embedding_data
        
        conn.execute(
            'INSERT INTO voice_profiles (name, embedding, first_seen, last_seen, total_recordings) '
            'VALUES (?, ?, ?, ?, ?)',
            (name, pickle.dumps(stored_data), now, now, 1)
        )
        pid = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.commit()
        conn.close()
        
        plda_manager.on_new_profile()
        wccn_manager.on_new_profile()
        score_norm_manager.on_new_profile()
        return pid
    except Exception as e:
        print(f"Error creating profile: {e}")
        return None


def update_voice_profile(profile_id, new_embedding_data):
    """Update profile with new embedding."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        row = conn.execute(
            'SELECT embedding, total_recordings FROM voice_profiles WHERE id=?',
            (profile_id,)
        ).fetchone()
        if not row:
            conn.close()
            return
        
        old_data = pickle.loads(row[0])
        total = row[1]
        
        quality = new_embedding_data.get('quality', 1.0) if isinstance(new_embedding_data, dict) else 1.0
        duration = new_embedding_data.get('duration', 1.0) if isinstance(new_embedding_data, dict) else 1.0
        
        if ENABLE_MULTI_CENTROID:
            if isinstance(old_data, dict) and 'centroids' in old_data:
                centroid_obj = VoiceProfileCentroid.from_dict(old_data)
            else:
                centroid_obj = VoiceProfileCentroid(max_centroids=MAX_CENTROIDS_PER_PROFILE)
                centroid_obj.add_embedding(old_data, 1.0, 1.0)
            
            centroid_obj.add_embedding(new_embedding_data, quality, duration)
            refined = centroid_obj.to_dict()
        else:
            alpha = max(ADAPTIVE_LEARNING_RATE[0],
                       min(1.0 / (total + 1), ADAPTIVE_LEARNING_RATE[1]))
            alpha *= quality
            
            old_r = _extract_resemblyzer(old_data)
            new_r = _extract_resemblyzer(new_embedding_data)
            refined_r = (1 - alpha) * old_r + alpha * new_r
            refined_r /= (np.linalg.norm(refined_r) + 1e-8)
            
            refined_e = None
            if ENABLE_ECAPA and isinstance(new_embedding_data, dict):
                old_e = old_data.get('ecapa_embedding') if isinstance(old_data, dict) else None
                new_e = new_embedding_data.get('ecapa_embedding')
                if old_e is not None and new_e is not None:
                    refined_e = (1 - alpha) * old_e + alpha * new_e
                    refined_e /= (np.linalg.norm(refined_e) + 1e-8)
                else:
                    refined_e = new_e if new_e is not None else old_e
            
            refined = {
                'resemblyzer_embedding': refined_r,
                'ecapa_embedding': refined_e,
                'quality': quality,
                'gender': new_embedding_data.get('gender', 'unknown') if isinstance(new_embedding_data, dict) else 'unknown',
            }
        
        now = datetime.now().isoformat()
        conn.execute(
            'UPDATE voice_profiles SET last_seen=?, total_recordings=total_recordings+1, embedding=? WHERE id=?',
            (now, pickle.dumps(refined), profile_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating profile: {e}")


# =========================================================================
# SEGMENT PROCESSING
# =========================================================================

def merge_close_segments(segments, gap_threshold=0.15):
    if not segments:
        return segments
    sorted_segs = sorted(segments, key=lambda x: x['start'])
    merged = [sorted_segs[0].copy()]
    for seg in sorted_segs[1:]:
        if seg['start'] - merged[-1]['end'] <= gap_threshold:
            merged[-1]['end'] = seg['end']
            merged[-1]['duration'] = merged[-1]['end'] - merged[-1]['start']
        else:
            merged.append(seg.copy())
    return merged


def process_speaker_parallel(args):
    speaker_label, speaker_info, audio_path = args
    segments = speaker_info['segments']
    total_duration = speaker_info['total_duration']
    
    merged = merge_close_segments(segments)
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
        res_vecs = np.array([e['resemblyzer_embedding'] for e in embeddings_data])
        median_emb = np.median(res_vecs, axis=0)
        dists = [np.linalg.norm(v - median_emb) for v in res_vecs]
        med_d = np.median(dists)
        keep = [(ed, w) for ed, w, d in zip(embeddings_data, weights, dists)
                if d < med_d * OUTLIER_REJECTION_FACTOR]
        if keep:
            embeddings_data, weights = zip(*keep)
            embeddings_data, weights = list(embeddings_data), list(weights)
    
    w_total = sum(weights)
    w_norm = [w / w_total for w in weights]
    avg_r = np.average([e['resemblyzer_embedding'] for e in embeddings_data],
                       axis=0, weights=w_norm)
    avg_r /= (np.linalg.norm(avg_r) + 1e-8)
    
    avg_e = None
    if ENABLE_ECAPA:
        ecapa_list = [(e['ecapa_embedding'], wn) for e, wn in zip(embeddings_data, w_norm)
                      if e.get('ecapa_embedding') is not None]
        if ecapa_list:
            vecs, ws = zip(*ecapa_list)
            ws_arr = np.array(ws); ws_arr /= ws_arr.sum()
            avg_e = np.average(list(vecs), axis=0, weights=ws_arr)
            avg_e /= (np.linalg.norm(avg_e) + 1e-8)
    
    avg_q = np.mean([e.get('quality', 1.0) for e in embeddings_data])
    genders = [e.get('gender', 'unknown') for e in embeddings_data]
    
    return {
        'label': speaker_label,
        'embedding': {
            'resemblyzer_embedding': avg_r,
            'ecapa_embedding': avg_e,
            'quality': float(avg_q),
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
        print(f"\n{'='*70}")
        print(f"🎙️  PROCESSING: {filename}")
        print(f"{'='*70}")
        t0 = time.time()
        
        speakers_data = perform_smart_diarization(audio_path)
        speakers = []
        
        if not speakers_data:
            print("⚠️ Single-speaker mode")
            emb_data = extract_voice_embedding(audio_path)
            if emb_data is None:
                return [{'speaker_index': 0, 'name': f"Unknown ({filename.split('.')[0]})",
                         'confidence': 0.0, 'voice_profile_id': None}]
            
            if emb_data.get('duration', 0) < MIN_PROFILE_DURATION:
                print(f"⚠️ Audio too short ({emb_data['duration']:.1f}s)")
                return [{'speaker_index': 0, 'name': f"Too Short ({filename.split('.')[0]})",
                         'confidence': 0.0, 'voice_profile_id': None}]
            
            match = find_matching_speaker(emb_data)
            if match:
                _print_match(match, emb_data)
                speakers.append({
                    'speaker_index': 0, 'name': match['name'],
                    'confidence': round(match['confidence'], 1),
                    'voice_profile_id': match['id'],
                })
                if match.get('verified') and match['similarity'] >= threshold_manager.strong_match_threshold:
                    update_voice_profile(match['id'], emb_data)
            else:
                name = f"Speaker {filename.split('.')[0]}"
                pid = create_voice_profile(name, emb_data)
                speakers.append({
                    'speaker_index': 0, 'name': name,
                    'confidence': 95.0, 'voice_profile_id': pid,
                })
        
        else:
            print(f"🎯 Multi-speaker processing: {len(speakers_data)} speakers")
            speaker_args = [
                (label, info, audio_path)
                for label, info in sorted(speakers_data.items(),
                                          key=lambda x: x[1]['total_duration'], reverse=True)
            ]
            max_w = min(len(speaker_args), multiprocessing.cpu_count())
            with ThreadPoolExecutor(max_workers=max_w) as ex:
                results = [f.result() for f in
                           {ex.submit(process_speaker_parallel, a): a[0]
                            for a in speaker_args}.keys()]
            results = sorted([r for r in results if r], key=lambda x: x['duration'], reverse=True)
            
            idx = 0
            for result in results:
                emb_data = result['embedding']
                if result['duration'] < MIN_PROFILE_DURATION:
                    print(f"  ⚠️ {result['label']}: {result['duration']:.1f}s – TOO SHORT")
                    continue
                
                _print_speaker_header(result, emb_data)
                match = find_matching_speaker(emb_data)
                
                if match:
                    num_cent = match.get('num_centroids', 1)
                    print(f"→ {match['name']} ({match['confidence']:.1f}%) [C={num_cent}]")
                    speakers.append({
                        'speaker_index': idx, 'name': match['name'],
                        'confidence': round(match['confidence'], 1),
                        'voice_profile_id': match['id'],
                    })
                    if match.get('verified') and match['similarity'] >= threshold_manager.strong_match_threshold:
                        update_voice_profile(match['id'], emb_data)
                else:
                    name = f"Speaker {idx + 1}"
                    print(f"→ NEW: {name}")
                    pid = create_voice_profile(name, emb_data)
                    speakers.append({
                        'speaker_index': idx, 'name': name,
                        'confidence': 95.0, 'voice_profile_id': pid,
                    })
                idx += 1
        
        elapsed = time.time() - t0
        print(f"\n{'='*70}")
        print(f"✅ COMPLETED: {len(speakers)} speaker(s) in {elapsed:.1f}s")
        print(f"{'='*70}\n")
        return speakers
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback; traceback.print_exc()
        return [{'speaker_index': 0, 'name': f"Error ({filename.split('.')[0]})",
                 'confidence': 0.0, 'voice_profile_id': None}]


def _model_tag(emb_data):
    has_ecapa = ENABLE_ECAPA and (emb_data.get('ecapa_embedding') is not None
                                  if isinstance(emb_data, dict) else False)
    plda_tag = "+PLDA" if plda_manager.is_ready else ""
    wccn_tag = "+WCCN" if wccn_manager.is_ready else ""
    snorm_tag = "+SN" if score_norm_manager.is_ready else ""
    adaptive_tag = "+AT" if ENABLE_ADAPTIVE_THRESHOLDS else ""
    centroid_tag = "+MC" if ENABLE_MULTI_CENTROID else ""
    
    if has_ecapa:
        return f"[E+R{plda_tag}{wccn_tag}{snorm_tag}{adaptive_tag}{centroid_tag}]"
    return f"[R{plda_tag}{wccn_tag}{snorm_tag}{adaptive_tag}{centroid_tag}]"


def _print_match(match, emb_data):
    tag = _model_tag(emb_data)
    g = f" [{emb_data.get('gender','?')}]" if ENABLE_GENDER_CLASSIFICATION else ""
    q = f" Q:{emb_data.get('quality',0)*100:.0f}%" if ENABLE_QUALITY_SCORING else ""
    num_cent = match.get('num_centroids', 1)
    c = f" [C={num_cent}]" if ENABLE_MULTI_CENTROID else ""
    print(f"✅ MATCHED: {match['name']} ({match['confidence']:.1f}%) {tag}{g}{q}{c}")


def _print_speaker_header(result, emb_data):
    tag = _model_tag(emb_data)
    g = f" [{emb_data.get('gender','?')}]" if ENABLE_GENDER_CLASSIFICATION else ""
    q = f" Q:{emb_data.get('quality',0)*100:.0f}%" if ENABLE_QUALITY_SCORING else ""
    print(f"  👤 {result['label']}: {result['duration']:.1f}s "
          f"({result['num_embeddings']} emb) {tag}{g}{q}", end=" ")


if __name__ == "__main__":
    print("\n🎯 Voice Recognition System  –  Score Normalization + WCCN + Multi-Centroid + PLDA")
    print("=" * 70)
    print(f"  Resemblyzer      : ✅")
    print(f"  ECAPA-TDNN       : {'✅' if ENABLE_ECAPA else '❌'}")
    print(f"  PLDA             : {'✅ trained' if plda_manager.is_ready else '⏳ waiting'}")
    print(f"  WCCN             : {'✅ trained' if wccn_manager.is_ready else '⏳ waiting'}")
    print(f"  Score Norm       : {'✅ (' + SCORE_NORM_METHOD + ')' if score_norm_manager.is_ready else '⏳ waiting'}")
    print(f"  Multi-Centroid   : {'✅ (max ' + str(MAX_CENTROIDS_PER_PROFILE) + ')' if ENABLE_MULTI_CENTROID else '❌'}")
    print(f"  Adaptive         : {'✅' if ENABLE_ADAPTIVE_THRESHOLDS else '❌'}")
    print("=" * 70)