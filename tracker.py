import numpy as np
from collections import defaultdict, deque
import cv2

class StableTracker:
    """Manages stable player and ball tracking with consistent IDs"""
    
    def __init__(self, history_length=30):
        self.history_length = history_length
        self.track_positions = defaultdict(lambda: deque(maxlen=history_length))
        self.track_velocities = defaultdict(lambda: deque(maxlen=history_length))
        self.track_classes = {}
        self.track_confidences = defaultdict(lambda: deque(maxlen=10))
        self.team_assignments = {}  # Stable team assignments
        self.last_seen = {}
        self.current_frame = 0
    
    def update(self, boxes, frame_shape, fps=30):
        """Update all tracks with new detections"""
        self.current_frame += 1
        current_tracks = {}
        
        if boxes is None or len(boxes) == 0:
            return current_tracks
        
        # First pass: collect all positions for team assignment
        positions = []
        valid_boxes = []
        
        for box in boxes:
            if box.id is not None:
                valid_boxes.append(box)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = float((x1 + x2) / 2)
                center_y = float((y1 + y2) / 2)
                positions.append((center_x, center_y, int(box.cls), int(box.id)))
        
        # Sort positions by x coordinate for team assignment
        positions.sort(key=lambda x: x[0])
        
        # Calculate team split point (midfield line)
        midfield_x = frame_shape[1] / 2
        
        # Process each box
        for box in valid_boxes:
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
            
            # Calculate velocity
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
            
            # Assign team for players
            if cls_id == 0:
                if track_id not in self.team_assignments:
                    # Use persistent team assignment based on historical position
                    team = 0 if center_x < midfield_x else 1
                    self.team_assignments[track_id] = team
                elif len(self.track_positions[track_id]) > 10:
                    # Re-evaluate team if player has moved significantly
                    avg_x = np.mean([p[0] for p in list(self.track_positions[track_id])[-10:]])
                    if avg_x < midfield_x - 50:
                        self.team_assignments[track_id] = 0
                    elif avg_x > midfield_x + 50:
                        self.team_assignments[track_id] = 1
                    # Keep current assignment if near midfield
            
            # Build current track info
            current_tracks[track_id] = {
                'position': (center_x, center_y),
                'bbox': bbox,
                'class': cls_id,
                'confidence': conf,
                'velocity': self.get_avg_velocity(track_id),
                'team': self.team_assignments.get(track_id, -1) if cls_id == 0 else -1
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
        """Check if track is stable"""
        return len(self.track_positions[track_id]) >= min_frames


class BallPossessionTracker:
    """IMPROVED: Tracks ball possession with better pass detection"""
    
    def __init__(self, proximity_threshold=80, history_length=60):
        self.proximity_threshold = proximity_threshold
        self.history_length = history_length
        self.possession_history = deque(maxlen=history_length)
        self.possession_durations = defaultdict(int)
        self.current_possessor = None
        self.last_possessor = None  # Last non-None possessor
        self.frames_since_change = 0
        self.frames_in_flight = 0  # Track how long ball has been in flight
        
    def update(self, ball_pos, player_tracks):
        """Update ball possession with improved tracking"""
        if ball_pos is None:
            self.possession_history.append(None)
            self.current_possessor = None
            self.frames_in_flight += 1
            return None
        
        closest_player = None
        min_distance = float('inf')
        
        # Find closest player to ball
        for track_id, data in player_tracks.items():
            player_pos = data['position']
            distance = np.sqrt(
                (ball_pos[0] - player_pos[0])**2 + 
                (ball_pos[1] - player_pos[1])**2
            )
            
            if distance < min_distance and distance < self.proximity_threshold:
                min_distance = distance
                closest_player = track_id
        
        # Update possession
        previous_possessor = self.current_possessor
        
        if closest_player is not None:
            # Someone has the ball
            self.current_possessor = closest_player
            self.last_possessor = closest_player
            self.possession_durations[closest_player] += 1
            self.frames_in_flight = 0
            
            if previous_possessor != closest_player:
                self.frames_since_change = 0
            else:
                self.frames_since_change += 1
        else:
            # Ball is in flight/transit
            self.current_possessor = None
            self.frames_in_flight += 1
            self.frames_since_change += 1
        
        self.possession_history.append(self.current_possessor)
        return self.current_possessor
    
    def get_potential_pass(self):
        """
        IMPROVED: Detect potential pass by looking at possession pattern
        Returns (from_player, to_player) if a pass is detected
        """
        if len(self.possession_history) < 5:
            return None
        
        # Get recent history (last 30 frames = 1 second)
        recent = list(self.possession_history)[-30:]
        
        # Find non-None values
        possessors = [p for p in recent if p is not None]
        
        if len(possessors) < 2:
            return None
        
        # Check if we have a clear transition from one player to another
        # Pattern: [P1, P1, ..., None, None, ..., P2, P2]
        from_player = possessors[0]
        to_player = possessors[-1]
        
        # Must be different players
        if from_player == to_player:
            return None
        
        # Check if recent possession shows this transition
        # Look for pattern where first half has from_player, second half has to_player
        first_half = possessors[:len(possessors)//2]
        second_half = possessors[len(possessors)//2:]
        
        # Count occurrences
        from_count_first = first_half.count(from_player)
        to_count_second = second_half.count(to_player)
        
        # If pattern is clear (dominant player in each half)
        if from_count_first >= len(first_half) * 0.6 and to_count_second >= len(second_half) * 0.6:
            return (from_player, to_player)
        
        return None
    
    def just_received_ball(self):
        """Check if current possessor just received the ball (within last 5 frames)"""
        return self.current_possessor is not None and self.frames_since_change < 5


class EventDetector:
    """IMPROVED: Detects football events with better pass detection"""
    
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.detected_passes = set()  # Track (from, to, frame_range) to avoid duplicates
        self.shot_cooldown = {}
        self.cooldown_frames = 30
        self.current_frame = 0
        self.last_pass_frame = 0
        
    def detect_pass(self, possession_tracker, player_tracks, ball_pos, ball_velocity, current_time):
        """
        IMPROVED: Detect pass with better logic
        """
        self.current_frame += 1
        
        if ball_pos is None:
            return None
        
        # Check if there's a potential pass pattern
        pass_info = possession_tracker.get_potential_pass()
        if pass_info is None:
            return None
        
        from_player, to_player = pass_info
        
        # Avoid detecting same pass multiple times
        frame_window = 20  # Only detect same pass once per 20 frames
        pass_signature = (from_player, to_player, self.current_frame // frame_window)
        
        if pass_signature in self.detected_passes:
            return None
        
        # Check if players are in tracks
        if from_player not in player_tracks or to_player not in player_tracks:
            return None
        
        # Calculate distance between players
        p1_pos = player_tracks[from_player]['position']
        p2_pos = player_tracks[to_player]['position']
        dist = np.sqrt((p1_pos[0] - p2_pos[0])**2 + (p1_pos[1] - p2_pos[1])**2)
        
        # Minimum pass distance (changed from 100 to 50 for shorter passes)
        if dist < 50:
            return None
        
        # Check minimum time between passes
        if self.current_frame - self.last_pass_frame < 10:
            return None
        
        # Register the pass
        self.detected_passes.add(pass_signature)
        self.last_pass_frame = self.current_frame
        
        # Clean old pass signatures
        if len(self.detected_passes) > 100:
            self.detected_passes.clear()
        
        return {
            'type': 'pass',
            'from_player': from_player,
            'to_player': to_player,
            'speed': float(ball_velocity),
            'distance': float(dist),
            'position': [float(ball_pos[0]), float(ball_pos[1])],
            'frame': self.current_frame,
            'time': float(current_time)
        }
    
    def detect_shot(self, possessor, ball_velocity, ball_pos, ball_direction, current_time):
        """Detect shot on goal"""
        if ball_velocity < 15.0:
            return None
        
        # Check if ball is moving towards goal
        if ball_direction[0] <= 0:
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
            'frame': self.current_frame,
            'time': float(current_time)
        }
    
    def calculate_goal_probability(self, ball_pos, ball_velocity, ball_direction, defenders_nearby):
        """Calculate probability of goal scoring"""
        h, w = self.frame_shape[:2]
        
        # Base probability from position
        x_normalized = ball_pos[0] / w
        if x_normalized < 0.5:
            prob = 5
        elif x_normalized < 0.7:
            prob = 20
        elif x_normalized < 0.85:
            prob = 45
        else:
            prob = 70
        
        # Angle to goal
        goal_center = (w, h / 2)
        dx = goal_center[0] - ball_pos[0]
        dy = goal_center[1] - ball_pos[1]
        
        if dx > 0:
            angle = abs(np.arctan2(dy, dx))
            angle_penalty = min(30, angle * 40)
            prob -= angle_penalty
        
        # Speed bonus
        if ball_velocity > 25:
            prob += 15
        elif ball_velocity > 15:
            prob += 8
        
        # Direction bonus
        if ball_direction[0] > 0:
            prob += 10
        
        # Defender pressure
        nearby_count = sum(1 for d in defenders_nearby if d < 50)
        prob -= nearby_count * 15
        
        return max(2, min(98, prob))
    
    def check_goal(self, ball_pos):
        """Check if ball crossed goal line"""
        h, w = self.frame_shape[:2]
        
        goal_y_min = h * 0.35
        goal_y_max = h * 0.65
        goal_x_threshold = w * 0.95
        
        if ball_pos[0] > goal_x_threshold:
            if goal_y_min < ball_pos[1] < goal_y_max:
                return True
        
        return False


class VisualRenderer:
    """IMPROVED: Handles all visual rendering with continuous DSS display"""
    
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
            thickness = 3
        elif action == 'pass':
            color = self.colors['pass']
            thickness = 2
        elif action == 'shot':
            color = self.colors['shot']
            thickness = 2
        else:
            team = data.get('team', -1)
            color = self.colors.get(f'team_{team}', (200, 200, 200))
            thickness = 2 if data['confidence'] > 0.5 else 1
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        team_marker = 'R' if data.get('team') == 0 else 'B' if data.get('team') == 1 else ''
        label = f"{team_marker}P{track_id}"
        if action:
            label += f" {action.upper()}"
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_ball(self, frame, data):
        """Draw ball with trail"""
        x1, y1, x2, y2 = data['bbox']
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        cv2.circle(frame, center, 8, self.colors['ball'], -1)
        cv2.circle(frame, center, 10, (255, 255, 255), 2)
    
    def draw_goal_area(self, frame):
        """Draw goal post area"""
        h, w = frame.shape[:2]
        
        goal_x = int(w * 0.95)
        goal_y1 = int(h * 0.35)
        goal_y2 = int(h * 0.65)
        
        color = (0, 255, 255)
        
        cv2.line(frame, (goal_x, goal_y1), (goal_x, goal_y2), color, 3)
        
        for i in range(goal_y1, goal_y2, 20):
            cv2.line(frame, (goal_x, i), (w, i), color, 1)
    
    def draw_stats_overlay(self, frame, stats):
        """Draw stats overlay at top"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        texts = [
            f"Goals: {stats.get('goals', 0)}",
            f"Shots: {stats.get('shots', 0)}",
            f"Passes: {stats.get('passes', 0)}",
            f"Time: {stats.get('time', '0.0')}s"
        ]
        
        x_offset = 20
        for text in texts:
            cv2.putText(frame, text, (x_offset, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            x_offset += 220
    
    def draw_goal_probability_bar(self, frame, probability):
        """Draw goal probability bar"""
        h, w = frame.shape[:2]
        
        bar_width = 300
        bar_height = 25
        bar_x = w - bar_width - 20
        bar_y = 100
        
        cv2.rectangle(frame, (bar_x - 3, bar_y - 3),
                     (bar_x + bar_width + 3, bar_y + bar_height + 3),
                     (255, 255, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (60, 60, 60), -1)
        
        fill_width = int(bar_width * (probability / 100))
        for i in range(fill_width):
            ratio = i / bar_width
            r = int(255 * ratio)
            g = int(255 * (1 - ratio))
            b = 0
            cv2.line(frame, (bar_x + i, bar_y),
                    (bar_x + i, bar_y + bar_height), (b, g, r), 1)
        
        cv2.putText(frame, "GOAL %", (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        pct_text = f"{probability:.0f}%"
        cv2.putText(frame, pct_text, (bar_x + bar_width//2 - 25, bar_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)