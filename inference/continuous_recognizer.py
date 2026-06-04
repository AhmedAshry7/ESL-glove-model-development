import sys
import os
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.pipeline_config as cfg
from recognition.activity_monitor import ActivityMonitor
from recognition.rolling_stats import RollingStatsBuffer
from recognition.sdtw_engine import SDTWEngine
from recognition.nms import NonMaximumSuppression
from features_engineering.template_manager import TemplateManager
from preprocessing.stream_preprocessor import normalize_frames

class ContinuousRecognizer:
    def __init__(self, model_dir: str):
        stats_path = os.path.join(model_dir, "templates/normalization_stats.npz")
        if os.path.exists(stats_path):
            data = np.load(stats_path)
            self.norm_stats = {'mean': data['mean'], 'std': data['std']}
        else:
            print("WARNING: normalization_stats.npz not found. Running without normalization.")
            self.norm_stats = None

        random_forest_path = os.path.join(model_dir, "prefilter_rf.joblib")
        if not os.path.exists(random_forest_path):
            raise FileNotFoundError(f"Prefilter model not found at {random_forest_path}")
        self.prefilter = joblib.load(random_forest_path)

        template_dir = os.path.join(model_dir, "templates")
        self.template_manager = TemplateManager(template_dir)
        self.templates = self.template_manager.get_templates()
        self.class_thresholds = self.template_manager.get_thresholds()
        self.n_classes = len(self.templates)

        discriminative_weights_path = os.path.join(template_dir, "discriminative_weights.npz")
        self.discriminative_weights = {}
        if os.path.exists(discriminative_weights_path):
            data = np.load(discriminative_weights_path)
            self.discriminative_weights = {k: data[k] for k in data.files}

        active_channels = np.where(cfg.CHANNEL_WEIGHTS > 0)[0]
        finger_indices = list(range(0, 16)) + list(range(28, 44))
        imu_indices    = list(range(16, 28)) + list(range(44, 56))

        active_finger_channels = [i for i in active_channels if i in finger_indices]
        active_imu_channels    = [i for i in active_channels if i in imu_indices]

        self.activity_monitor = ActivityMonitor(
            finger_thresh=cfg.FINGER_IDLE_THRESHOLD_NORM,
            arm_thresh=cfg.ARM_IDLE_THRESHOLD,
            idle_frames_req=cfg.IDLE_FRAMES_REQUIRED,
            active_finger_channels=active_finger_channels,
            active_imu_channels=active_imu_channels
        )
        self.rolling_stats = RollingStatsBuffer(
            window_size=cfg.ROLLING_WINDOW_SIZE,
            n_features=cfg.N_FEATURES_PER_FRAME
        )
        self.sdtw_engine = SDTWEngine(
            templates=self.templates,
            weights=cfg.CHANNEL_WEIGHTS,
            config=cfg,
            class_thresholds=self.class_thresholds
        )
        self.nms = NonMaximumSuppression(
            overlap_threshold=cfg.NMS_OVERLAP_THRESHOLD,
            cooldown_frames=cfg.NMS_COOLDOWN_FRAMES
        )

        self.frame_count = 0
        self.was_active = False
        self.current_candidates = []
        self.current_probabilitys = []
        self.frame_history = {}

    def _reset_state(self):
        self.activity_monitor.prev_frame = None
        self.activity_monitor.idle_counter = 0
        self.activity_monitor.is_active = False
        self.rolling_stats.idx = 0
        self.rolling_stats.full = False
        self.sdtw_engine.reset_all()
        self.nms = NonMaximumSuppression(
            overlap_threshold=cfg.NMS_OVERLAP_THRESHOLD,
            cooldown_frames=cfg.NMS_COOLDOWN_FRAMES
        )
        self.frame_count = 0
        self.was_active = False
        self.current_candidates = []
        self.current_probabilitys = []
        self.frame_history = {}

    def feed_frame(self, frame: np.ndarray) -> list:
        self.frame_count += 1

        if self.norm_stats is not None:
            frame = normalize_frames(frame.reshape(1, -1), self.norm_stats)[0]

        self.frame_history[self.frame_count] = frame
        if self.frame_count - 300 in self.frame_history:
            del self.frame_history[self.frame_count - 300]

        is_active = self.activity_monitor.feed(frame)
        window_frames = int(cfg.NMS_WINDOW_SECONDS * cfg.TARGET_SAMPLE_HZ)
                
        if not is_active:
            if self.was_active:
                # Transition ACTIVE → IDLE: Force emit any detections stuck in a valley
                forced_detections = self.sdtw_engine.force_emit_all(self.frame_count)
                # Gate before NMS to avoid triggering cooldown for weak false positives
                forced_detections = [d for d in forced_detections if (1.0 - d['distance']) >= cfg.CONFIDENCE_THRESHOLD]

                emissions = self.nms.process_detections(
                    forced_detections, self.frame_count, window_frames, 
                    self.frame_history, self.discriminative_weights, self.templates, force_flush=True
                )
            else:
                emissions = self.nms.process_detections(
                    [], self.frame_count, window_frames, 
                    self.frame_history, self.discriminative_weights, self.templates, force_flush=True
                )
                
            self.was_active = False
            final_emissions = [e for e in emissions if (1.0 - e['distance']) >= cfg.CONFIDENCE_THRESHOLD]
            return final_emissions

        self.was_active = True

        self.rolling_stats.push(frame)

        #pre-filter to run after the buffer is filled
        if self.frame_count % cfg.PREFILTER_INTERVAL == 0:
            # Gate: don't classify until enough history is accumulated
            if self.rolling_stats.fill_ratio >= cfg.PREFILTER_MIN_FILL_RATIO:
                features = self.rolling_stats.calc_features()
                if features is not None:
                    probability = self.prefilter.predict_proba(features.reshape(1, -1))[0]
                    top_k = cfg.get_top_k(self.n_classes)
                    top_indices = np.argsort(probability)[-top_k:][::-1]
                    top_candidates = [self.prefilter.classes_[i] for i in top_indices]

                    self.current_candidates = top_candidates
                    self.current_probabilitys = [probability[i] for i in top_indices]
                    self.sdtw_engine.update_candidates(top_candidates)

        new_detections = self.sdtw_engine.feed(frame, self.frame_count)
        # Gate before NMS to avoid triggering cooldown for weak false positives
        new_detections = [d for d in new_detections if (1.0 - d['distance']) >= cfg.CONFIDENCE_THRESHOLD]


        emissions = self.nms.process_detections(
            new_detections, self.frame_count, window_frames,
            self.frame_history, self.discriminative_weights, self.templates
        )

        final_emissions = [
            e for e in emissions
            if (1.0 - e['distance']) >= cfg.CONFIDENCE_THRESHOLD
        ]

        return final_emissions
