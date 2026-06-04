class NonMaximumSuppression:
    """
    Resolves overlapping detections and enforces a per-class cooldown
    to avoid duplicate emissions for the same sign.
    """
    def __init__(self, overlap_threshold=0.5, cooldown_frames=25):
        self.overlap_threshold = overlap_threshold
        self.cooldown_frames = cooldown_frames
        
        self.recent_detections = []  # List of dicts
        self.last_global_emission = -9999  # global_frame
        self.pending_emissions = []
        
    def process_detections(self, new_detections: list, current_frame: int, window_frames=150, frame_history: dict = None, disc_weights: dict = None, templates: dict = None, force_flush: bool = False) -> list:
        """
        Takes detections from current frame, adds to sliding window,
        resolves overlaps, and returns any final emitted detections.
        """
        # Add new detections to window
        self.recent_detections.extend(new_detections)
        
        # Prune old detections outside window
        self.recent_detections = [
            d for d in self.recent_detections 
            if current_frame - d['global_end_frame'] <= window_frames
        ]
        
        if not self.recent_detections:
            return []
            
        # Sort detections by normalized distance (lowest is best)
        sorted_dets = sorted(self.recent_detections, key=lambda x: x['distance'])
        
        emitted = []
        accepted = []
        
        for det in sorted_dets:
            # Check overlap with already accepted detections in this pass
            overlap = False
            for i, acc in enumerate(accepted):
                # Calculate Intersection over Union (IoU) of time ranges
                start1, end1 = det['global_start_frame_est'], det['global_end_frame']
                start2, end2 = acc['global_start_frame_est'], acc['global_end_frame']
                
                intersection = max(0, min(end1, end2) - max(start1, start2))
                union = max(end1, end2) - min(start1, start2)
                
                if union == 0:
                    union = 1
                    
                if intersection / union > self.overlap_threshold:
                    overlap = True
                    
                    # ── Tie-Breaker Logic ──
                    # If this discarded detection is within 20% of the accepted one,
                    # and they are different classes, use discriminative weights to re-score
                    if abs(det['distance'] - acc['distance']) / max(0.001, acc['distance']) < 0.20:
                        lA, lB = acc['label'], det['label']
                        if lA != lB and disc_weights and templates and frame_history:
                            key = f"{lA}_vs_{lB}"
                            if key in disc_weights:
                                W_AB = disc_weights[key]
                                
                                # Extract frames
                                fA = [frame_history[f] for f in range(acc['global_start_frame_est'], acc['global_end_frame']+1) if f in frame_history]
                                fB = [frame_history[f] for f in range(det['global_start_frame_est'], det['global_end_frame']+1) if f in frame_history]
                                
                                if len(fA) > 0 and len(fB) > 0:
                                    tmpl_A = templates[lA][acc['template_idx']]
                                    tmpl_B = templates[lB][det['template_idx']]
                                    
                                    from recognition.sdtw_engine import compute_static_dtw
                                    import numpy as np
                                    
                                    cost_A = compute_static_dtw(np.array(fA), tmpl_A, W_AB)
                                    cost_B = compute_static_dtw(np.array(fB), tmpl_B, W_AB)
                                    
                                    if cost_B < cost_A:
                                        # The new detection is actually better! Swap them.
                                        accepted[i] = det
                                        
                    break # We already found an overlap, don't check further
                    
            if not overlap:
                accepted.append(det)
                
        # Emit a detection if it ends exactly at current_frame 
        # AND it passes the cooldown check.
        # We need a buffering system to prevent early false positives from locking out true positives.
        
        # 1. Add accepted detections to a pending buffer
        for det in accepted:
            if det['global_end_frame'] == current_frame:
                self.pending_emissions.append(det)
                
        emitted = []
        
        # 2. If we have pending emissions, wait until N frames have passed since the FIRST pending emission
        if self.pending_emissions:
            # Sort pending by distance (lowest first)
            self.pending_emissions.sort(key=lambda x: x['distance'])
            
            best_det = self.pending_emissions[0]
            first_det_frame = min(d['global_end_frame'] for d in self.pending_emissions)
            
            # Wait 10 frames (200ms) after the first candidate to see if a better one finishes
            if force_flush or current_frame - first_det_frame >= 10:
                if current_frame - self.last_global_emission >= self.cooldown_frames:
                    emitted.append(best_det)
                    self.last_global_emission = current_frame
                # Clear pending buffer after the window elapses
                self.pending_emissions = []
                
        return emitted
