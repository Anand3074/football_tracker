"""
Improved tracking system with stable IDs and smooth detection
"""
import numpy as np
from collections import defaultdict, deque
import cv2


class StableTracker:
    """Manages stable player and ball tracking with consistent IDs"""
    
    def __init__(self, history_length=30):
        self.history_length = history_length
        self.track_positions = defaultdict(lambda: deque(maxlen=history_length))
        self.track_velocities = defaultdict(lambda: deque(maxlen=history_length))
        self.track_classes = {}  # Store class for each track_id
        self.track_confidences = defaultdict(lambda: deque(maxlen=10))
        self.team_assignments = {}  # Stable team assignments
        self.last_seen = {}  # Frame number when track was last seen
        self.current_frame = 0
        
    def update(self, boxes, frame_shape, fps=30):
        """Update all tracks with new detections"""
        self.current_frame += 1
        current_tracks = {}
        
        if boxes is None or len(boxes) == 0:
            return current_tracks
            
        for box in boxes:
            track_id = int(box.id)
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate center
            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            
            # Store track info
            self.track_classes[track_id] = cls_id
            self.last_seen[track_id] = self.current_frame
            self.track_confidences[track_id].append(conf)
            
            # Calculate velocity using smoothed positions
            prev_positions = list(self.track_positions[track_id])
            if len(prev_positions) > 0:
                prev_x, prev_y = prev_positions[-1]
                dx = center_x - prev_x
                dy = center_y - prev_y
                velocity = np.sqrt(dx**2 + dy**2) * fps
                self.track_velocities[track_id].append(velocity)
            else:
                self.track_velocities[track_id].append(0.0)
            
            # Update position history
            self.track_positions[track_id].append((center_x, center_y))
            
            # Assign team for players on first detection
            if cls_id == 0 and track_id not in self.team_assignments:
                # Simple team assignment based on horizontal position
                self.team_assignments[track_id] = 0 if center_x < frame_shape[1] / 2 else 1
            
            # Build current track info
            current_tracks[track_id] = {
                'position': (center_x, center_y),
                'bbox': bbox,
                'class': cls_id,
                'confidence': conf,
                'velocity': self.get_avg_velocity(track_id),
                'team': self.team_assignments.get(track_id, -1)
            }
        
        return current_tracks
    
    def get_smoothed_position(self, track_id, window=5):
        """Get smoothed position using moving average"""
        positions = list(self.track_positions[track_id])
        if len(positions) < window:
            return positions[-1] if positions else None
        
        recent = positions[-window:]
        avg_x = sum(p[0] for p in recent) / len(recent)
        avg_y = sum(p[1] for p in recent) / len(recent)
        return (avg_x, avg_y)
    
    def get_avg_velocity(self, track_id, window=5):
        """Get average velocity over window"""
        velocities = list(self.track_velocities[track_id])
        if not velocities:
            return 0.0
        recent = velocities[-window:]
        return sum(recent) / len(recent)
    
    def get_avg_confidence(self, track_id):
        """Get average confidence for track"""
        confs = list(self.track_confidences[track_id])
        return sum(confs) / len(confs) if confs else 0.0
    
    def is_stable(self, track_id, min_frames=5):
        """Check if track is stable (seen for minimum frames)"""
        return len(self.track_positions[track_id]) >= min_frames


class BallPossessionTracker:
    """Tracks which player has ball possession"""
    
    def __init__(self, proximity_threshold=80, history_length=15):
        self.proximity_threshold = proximity_threshold
        self.history_length = history_length
        self.possession_history = deque(maxlen=history_length)
        self.possession_durations = defaultdict(int)
        self.current_possessor = None
        
    def update(self, ball_pos, player_tracks):
        """Update ball possession based on proximity"""
        if ball_pos is None:
            self.possession_history.append(None)
            return None
        
        closest_player = None
        min_distance = float('inf')
        
        # Find closest player to ball
        for track_id, data in player_tracks.items():
            if data['class'] != 0:  # Skip non-players
                continue
                
            player_pos = data['position']
            distance = np.sqrt(
                (ball_pos[0] - player_pos[0])**2 + 
                (ball_pos[1] - player_pos[1])**2
            )
            
            if distance < min_distance and distance < self.proximity_threshold:
                min_distance = distance
                closest_player = track_id
        
        # Update possession
        self.current_possessor = closest_player
        self.possession_history.append(closest_player)
        
        if closest_player is not None:
            self.possession_durations[closest_player] += 1
        
        return closest_player
    
    def get_previous_possessor(self, frames_back=3):
        """Get who had possession N frames ago"""
        history = list(self.possession_history)
        if len(history) < frames_back + 1:
            return None
        return history[-(frames_back + 1)]
    
    def possession_changed(self):
        """Check if possession just changed"""
        history = list(self.possession_history)
        if len(history) < 2:
            return False
        return history[-1] != history[-2] and history[-1] is not None and history[-2] is not None


class EventDetector:
    """Detects football events: passes, shots, goals"""
    
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.pass_cooldown = {}  # Prevent duplicate pass detection
        self.shot_cooldown = {}  # Prevent duplicate shot detection
        self.cooldown_frames = 15
        self.current_frame = 0
        
    def detect_pass(self, possessor, previous_possessor, ball_velocity, ball_pos):
        """Detect completed pass between players"""
        self.current_frame += 1
        
        if possessor is None or previous_possessor is None:
            return None
        
        if possessor == previous_possessor:
            return None
        
        # Check cooldown to prevent duplicate detections
        cooldown_key = (previous_possessor, possessor)
        if cooldown_key in self.pass_cooldown:
            if self.current_frame - self.pass_cooldown[cooldown_key] < self.cooldown_frames:
                return None
        
        # Pass criteria: ball moved with sufficient speed
        if ball_velocity > 8.0:  # Minimum speed threshold
            self.pass_cooldown[cooldown_key] = self.current_frame
            
            return {
                'type': 'pass',
                'from_player': previous_possessor,
                'to_player': possessor,
                'speed': float(ball_velocity),
                'position': [float(ball_pos[0]), float(ball_pos[1])],
                'frame': self.current_frame
            }
        
        return None
    
    def detect_shot(self, possessor, ball_velocity, ball_pos, ball_direction):
        """Detect shot on goal"""
        if ball_velocity < 15.0:  # Minimum shot speed
            return None
        
        # Check if ball is moving towards goal (right side of field)
        if ball_direction[0] <= 0:  # Not moving towards goal
            return None
        
        # Check if in attacking third
        if ball_pos[0] < self.frame_shape[1] * 0.6:
            return None
        
        # Check cooldown
        if possessor in self.shot_cooldown:
            if self.current_frame - self.shot_cooldown[possessor] < self.cooldown_frames * 2:
                return None
        
        self.shot_cooldown[possessor] = self.current_frame
        
        return {
            'type': 'shot',
            'player': possessor,
            'speed': float(ball_velocity),
            'position': [float(ball_pos[0]), float(ball_pos[1])],
            'direction': [float(ball_direction[0]), float(ball_direction[1])],
            'frame': self.current_frame
        }
    
    def calculate_goal_probability(self, ball_pos, ball_velocity, ball_direction, defenders_nearby):
        """Calculate probability of goal scoring"""
        h, w = self.frame_shape[:2]
        
        # Base probability from position
        x_normalized = ball_pos[0] / w
        if x_normalized < 0.5:
            prob = 5  # Midfield
        elif x_normalized < 0.7:
            prob = 20  # Approaching
        elif x_normalized < 0.85:
            prob = 45  # Attacking third
        else:
            prob = 70  # Very close
        
        # Angle to goal center
        goal_center = (w, h / 2)
        dx = goal_center[0] - ball_pos[0]
        dy = goal_center[1] - ball_pos[1]
        
        if dx > 0:  # Ball before goal
            angle = abs(np.arctan2(dy, dx))
            angle_penalty = min(30, angle * 40)
            prob -= angle_penalty
        
        # Speed bonus
        if ball_velocity > 25:
            prob += 15
        elif ball_velocity > 15:
            prob += 8
        
        # Direction bonus (moving towards goal)
        if ball_direction[0] > 0:
            prob += 10
        
        # Defender pressure
        nearby_count = sum(1 for d in defenders_nearby if d < 50)
        prob -= nearby_count * 15
        
        return max(2, min(98, prob))
    
    def check_goal(self, ball_pos):
        """Check if ball crossed goal line"""
        h, w = self.frame_shape[:2]
        
        # Goal area (right side, centered vertically)
        goal_y_min = h * 0.35
        goal_y_max = h * 0.65
        goal_x_threshold = w * 0.95
        
        if ball_pos[0] > goal_x_threshold:
            if goal_y_min < ball_pos[1] < goal_y_max:
                return True
        
        return False


class VisualRenderer:
    """Handles all visual rendering with smooth annotations"""
    
    def __init__(self):
        self.colors = {
            'team_0': (100, 100, 255),  # Red team (BGR)
            'team_1': (255, 100, 100),  # Blue team (BGR)
            'ball': (0, 255, 255),      # Yellow
            'possession': (0, 255, 0),  # Green
            'pass': (255, 255, 0),      # Cyan
            'shot': (100, 100, 255)     # Red
        }
        
    def draw_player(self, frame, track_id, data, has_possession=False, action=None):
        """Draw player bounding box and label"""
        x1, y1, x2, y2 = data['bbox']
        
        # Choose color
        if has_possession:
            color = self.colors['possession']
        elif action == 'pass':
            color = self.colors['pass']
        elif action == 'shot':
            color = self.colors['shot']
        else:
            team = data.get('team', -1)
            color = self.colors.get(f'team_{team}', (200, 200, 200))
        
        # Draw box with thickness based on confidence
        thickness = 2 if data['confidence'] > 0.5 else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        team_emoji = '🔴' if data.get('team') == 0 else '🔵' if data.get('team') == 1 else ''
        label = f"{team_emoji}P{track_id}"
        if action:
            label += f" {action.upper()}"
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_ball(self, frame, data):
        """Draw ball with trail"""
        x1, y1, x2, y2 = data['bbox']
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Draw circle
        cv2.circle(frame, center, 8, self.colors['ball'], -1)
        cv2.circle(frame, center, 10, (255, 255, 255), 2)
    
    def draw_goal_area(self, frame):
        """Draw goal post area"""
        h, w = frame.shape[:2]
        
        # Goal coordinates
        goal_x = int(w * 0.95)
        goal_y1 = int(h * 0.35)
        goal_y2 = int(h * 0.65)
        
        color = (0, 255, 255)  # Yellow
        
        # Draw goal posts
        cv2.line(frame, (goal_x, goal_y1), (goal_x, goal_y2), color, 3)
        
        # Draw net pattern
        for i in range(goal_y1, goal_y2, 20):
            cv2.line(frame, (goal_x, i), (w, i), color, 1)
    
    def draw_stats_overlay(self, frame, stats):
        """Draw stats overlay at top"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw stats
        texts = [
            f"⚽ Goals: {stats.get('goals', 0)}",
            f"🎯 Shots: {stats.get('shots', 0)}",
            f"🔄 Passes: {stats.get('passes', 0)}",
            f"⏱️ {stats.get('time', '0.0')}s"
        ]
        
        x_offset = 20
        for text in texts:
            cv2.putText(frame, text, (x_offset, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            x_offset += 250
    
    def draw_goal_probability_bar(self, frame, probability):
        """Draw prominent goal probability bar"""
        h, w = frame.shape[:2]
        
        # Bar dimensions
        bar_width = 400
        bar_height = 35
        bar_x = w - bar_width - 30
        bar_y = 120
        
        # Background
        cv2.rectangle(frame, (bar_x - 5, bar_y - 5),
                     (bar_x + bar_width + 5, bar_y + bar_height + 5),
                     (255, 255, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (60, 60, 60), -1)
        
        # Filled portion with gradient
        fill_width = int(bar_width * (probability / 100))
        for i in range(fill_width):
            ratio = i / bar_width
            # Green to red gradient
            r = int(255 * ratio)
            g = int(255 * (1 - ratio))
            b = 0
            cv2.line(frame, (bar_x + i, bar_y),
                    (bar_x + i, bar_y + bar_height), (b, g, r), 1)
        
        # Label and percentage
        label = "GOAL PROBABILITY"
        cv2.putText(frame, label, (bar_x, bar_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        pct_text = f"{probability:.0f}%"
        cv2.putText(frame, pct_text, (bar_x + bar_width//2 - 30, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)