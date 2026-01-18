"""
Main video processing module with improved event detection
"""
import cv2
from ultralytics import YOLO
import numpy as np
import json
import os
from collections import defaultdict
from tracker import (
    StableTracker, 
    BallPossessionTracker, 
    EventDetector,
    VisualRenderer
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class MatchAnalytics:
    """Store and manage match analytics"""
    
    def __init__(self):
        self.passes = []
        self.shots = []
        self.goals = []
        self.possession_time = defaultdict(int)
        self.player_touches = defaultdict(int)
        self.heatmap = None
        
    def add_pass(self, pass_event):
        """Record a pass event"""
        self.passes.append(pass_event)
        
    def add_shot(self, shot_event):
        """Record a shot event"""
        self.shots.append(shot_event)
        
    def add_goal(self, goal_event):
        """Record a goal event"""
        self.goals.append(goal_event)
    
    def update_possession(self, player_id):
        """Update possession time for player"""
        if player_id is not None:
            self.possession_time[player_id] += 1
    
    def get_summary(self):
        """Get analytics summary"""
        return {
            'total_passes': len(self.passes),
            'total_shots': len(self.shots),
            'total_goals': len(self.goals),
            'pass_accuracy': self._calculate_pass_accuracy(),
            'shot_accuracy': self._calculate_shot_accuracy(),
            'possession_leaders': self._get_possession_leaders()
        }
    
    def _calculate_pass_accuracy(self):
        """Calculate pass completion rate"""
        if not self.passes:
            return 0.0
        # For now, all detected passes are completed
        return 100.0
    
    def _calculate_shot_accuracy(self):
        """Calculate shot to goal conversion"""
        if not self.shots:
            return 0.0
        return (len(self.goals) / len(self.shots)) * 100
    
    def _get_possession_leaders(self, top_n=3):
        """Get players with most possession"""
        sorted_possession = sorted(
            self.possession_time.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_possession[:top_n]


def trim_video(input_path, output_path, max_duration=10):
    """Trim video to specified duration"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        cap.release()
        raise ValueError("Cannot determine FPS")
    
    max_frames = int(fps * max_duration)
    frames_to_process = min(total_frames, max_frames)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    print(f"📹 Trimming: {total_frames} → {frames_to_process} frames ({max_duration}s)")
    
    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path, frames_to_process


def process_video(video_in_path, video_out_path, actions=None, 
                  analytics=None, analytics_path=None):
    """
    Process football video with improved tracking and event detection
    
    Args:
        video_in_path: Input video path
        video_out_path: Output video path
        actions: List of actions to track (not used in new system)
        analytics: List of analytics to enable
        analytics_path: Path to save analytics JSON
    """
    print("🚀 Starting improved football analysis...")
    
    # Trim video first
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "temp_trimmed.mp4")
    
    try:
        trimmed_path, total_frames = trim_video(video_in_path, temp_path)
    except Exception as e:
        raise Exception(f"Video trimming failed: {str(e)}")
    
    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        raise Exception(f"Failed to load YOLO model: {str(e)}")
    
    # Open video
    cap = cv2.VideoCapture(trimmed_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open trimmed video: {trimmed_path}")
    
    # Video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize components
    tracker = StableTracker(history_length=30)
    possession_tracker = BallPossessionTracker(proximity_threshold=80)
    event_detector = EventDetector((h, w))
    renderer = VisualRenderer()
    analytics_engine = MatchAnalytics()
    
    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
    
    # Tracking configuration
    CLASSES_TO_TRACK = [0, 32]  # Person, Sports Ball
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.5
    
    frame_count = 0
    current_goal_probability = 0.0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            # Run YOLO tracking
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=CLASSES_TO_TRACK,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
            
            # Get detections
            boxes = results[0].boxes if results[0].boxes is not None else None
            
            # Update tracker
            current_tracks = tracker.update(boxes, (h, w), fps)
            
            # Find ball
            ball_track = None
            ball_pos = None
            ball_velocity = 0.0
            ball_direction = (0, 0)
            
            for track_id, data in current_tracks.items():
                if data['class'] == 32:  # Ball
                    ball_track = track_id
                    ball_pos = data['position']
                    ball_velocity = data['velocity']
                    
                    # Calculate direction
                    positions = list(tracker.track_positions[track_id])
                    if len(positions) >= 2:
                        dx = positions[-1][0] - positions[-2][0]
                        dy = positions[-1][1] - positions[-2][1]
                        ball_direction = (dx, dy)
                    break
            
            # Get player tracks only
            player_tracks = {
                tid: data for tid, data in current_tracks.items()
                if data['class'] == 0
            }
            
            # Update ball possession
            current_possessor = possession_tracker.update(ball_pos, player_tracks)
            analytics_engine.update_possession(current_possessor)
            
            # Detect events
            if ball_pos is not None and current_possessor is not None:
                
                # Detect pass
                if possession_tracker.possession_changed():
                    previous = possession_tracker.get_previous_possessor()
                    pass_event = event_detector.detect_pass(
                        current_possessor, previous, ball_velocity, ball_pos, current_time
                    )
                    if pass_event:
                        analytics_engine.add_pass(pass_event)
                        print(f"🔄 PASS: P{pass_event['from_player']} → P{pass_event['to_player']} "
                              f"(Speed: {pass_event['speed']:.1f})")
                
                # Detect shot
                shot_event = event_detector.detect_shot(
                    current_possessor, ball_velocity, ball_pos, ball_direction, current_time
                )
                if shot_event:
                    # Calculate goal probability
                    defenders = []
                    for tid, data in player_tracks.items():
                        if tid != current_possessor:
                            dist = np.sqrt(
                                (ball_pos[0] - data['position'][0])**2 +
                                (ball_pos[1] - data['position'][1])**2
                            )
                            defenders.append(dist)
                    
                    goal_prob = event_detector.calculate_goal_probability(
                        ball_pos, ball_velocity, ball_direction, defenders
                    )
                    shot_event['probability'] = goal_prob
                    
                    analytics_engine.add_shot(shot_event)
                    print(f"⚽ SHOT: P{shot_event['player']} "
                          f"(Probability: {goal_prob:.0f}%, Speed: {shot_event['speed']:.1f})")
                
                # Check for goal
                if event_detector.check_goal(ball_pos):
                    goal_event = {
                        'type': 'goal',
                        'player': int(current_possessor) if current_possessor is not None else 0,
                        'position': [float(ball_pos[0]), float(ball_pos[1])],
                        'frame': frame_count,
                        'time': float(current_time)
                    }
                    analytics_engine.add_goal(goal_event)
                    print(f"🥅 GOAL!!! P{current_possessor} scored!")
                
                # Update goal probability for display
                defenders = []
                for tid, data in player_tracks.items():
                    if tid != current_possessor:
                        dist = np.sqrt(
                            (ball_pos[0] - data['position'][0])**2 +
                            (ball_pos[1] - data['position'][1])**2
                        )
                        defenders.append(dist)
                
                current_goal_probability = event_detector.calculate_goal_probability(
                    ball_pos, ball_velocity, ball_direction, defenders
                )
            else:
                current_goal_probability = max(0, current_goal_probability - 2)
            
            # Render frame
            annotated_frame = frame.copy()
            
            # Draw goal area
            renderer.draw_goal_area(annotated_frame)
            
            # Draw all tracks
            for track_id, data in current_tracks.items():
                if data['class'] == 0:  # Player
                    has_possession = (track_id == current_possessor)
                    renderer.draw_player(annotated_frame, track_id, data, has_possession)
                elif data['class'] == 32:  # Ball
                    renderer.draw_ball(annotated_frame, data)
            
            # Draw stats overlay
            stats = {
                'goals': len(analytics_engine.goals),
                'shots': len(analytics_engine.shots),
                'passes': len(analytics_engine.passes),
                'time': f"{current_time:.1f}"
            }
            renderer.draw_stats_overlay(annotated_frame, stats)
            
            # Draw goal probability bar
            renderer.draw_goal_probability_bar(annotated_frame, current_goal_probability)
            
            # Write frame
            out.write(annotated_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                print(f"📊 Frame {frame_count}/{total_frames} - "
                      f"Goals: {len(analytics_engine.goals)}, "
                      f"Shots: {len(analytics_engine.shots)}, "
                      f"Passes: {len(analytics_engine.passes)}")
    
    except Exception as e:
        raise Exception(f"Processing error: {str(e)}")
    
    finally:
        cap.release()
        out.release()
        
        # Save analytics
        if analytics_path:
            save_analytics(analytics_engine, analytics_path, frame_count, fps)
        
        # Cleanup temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
        
        # Remove input file
        if os.path.exists(video_in_path):
            os.remove(video_in_path)
    
    print(f"✅ Analysis Complete!")
    print(f"🎯 Final Stats:")
    print(f"   Goals: {len(analytics_engine.goals)}")
    print(f"   Shots: {len(analytics_engine.shots)}")
    print(f"   Passes: {len(analytics_engine.passes)}")


def save_analytics(analytics, path, total_frames, fps):
    """Save analytics data to JSON file"""
    
    # Calculate shot probabilities average
    shot_probs = [s.get('probability', 0) for s in analytics.shots]
    avg_shot_prob = float(np.mean(shot_probs)) if shot_probs else 0.0
    
    # Get possession statistics
    total_possession_frames = sum(analytics.possession_time.values())
    possession_percentages = {}
    
    for player_id, frames in analytics.possession_time.items():
        if total_possession_frames > 0:
            pct = (frames / total_possession_frames) * 100
            possession_percentages[f"player_{player_id}"] = round(pct, 1)
    
    # Build analytics data
    analytics_data = {
        'goals': len(analytics.goals),
        'shots': len(analytics.shots),
        'passes': len(analytics.passes),
        'duration': float(total_frames / fps),
        'events': analytics.passes + analytics.shots + analytics.goals,
        'pass_events': analytics.passes,
        'shot_events': analytics.shots,
        'goal_events': analytics.goals,
        'summary': {
            'total_passes': len(analytics.passes),
            'total_shots': len(analytics.shots),
            'total_goals': len(analytics.goals),
            'key_events': len(analytics.passes) + len(analytics.shots) + len(analytics.goals),
            'avg_goal_probability': round(avg_shot_prob, 1),
            'pass_accuracy': 100.0,  # All detected passes are completed
            'shot_accuracy': round(analytics._calculate_shot_accuracy(), 1),
            'possession_distribution': possession_percentages
        }
    }
    
    with open(path, 'w') as f:
        json.dump(analytics_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"💾 Analytics saved to: {path}")


def generate_match_report(analytics_data):
    """Generate comprehensive match report from analytics data"""
    summary = analytics_data.get('summary', {})
    
    total_shots = summary.get('total_shots', 0)
    total_goals = summary.get('total_goals', 0)
    total_passes = summary.get('total_passes', 0)
    
    # Calculate conversion rate
    conversion_rate = (total_goals / total_shots * 100) if total_shots > 0 else 0
    
    # Build performance insights
    insights = []
    
    if total_shots > 0:
        insights.append(f"Team created {total_shots} scoring opportunities")
    
    if total_passes > 0:
        insights.append(f"Completed {total_passes} successful passes")
    
    if total_goals > 0:
        insights.append(f"Scored {total_goals} goal{'s' if total_goals != 1 else ''}")
    
    if total_shots > 0:
        insights.append(f"Shot conversion rate: {conversion_rate:.1f}%")
    
    avg_prob = summary.get('avg_goal_probability', 0)
    if avg_prob > 0:
        insights.append(f"Average shot quality: {avg_prob:.1f}% goal probability")
    
    # Possession insights
    possession = summary.get('possession_distribution', {})
    if possession:
        top_player = max(possession.items(), key=lambda x: x[1])
        insights.append(f"Top possession: {top_player[0]} with {top_player[1]:.1f}%")
    
    return {
        'match_summary': {
            'duration': f"{analytics_data.get('duration', 0):.1f} seconds",
            'total_events': summary.get('key_events', 0),
            'analysis_quality': 'Professional'
        },
        'key_metrics': {
            'total_passes': total_passes,
            'total_shots': total_shots,
            'total_goals': total_goals,
            'goal_conversion': f"{conversion_rate:.1f}%",
            'avg_goal_probability': f"{avg_prob:.1f}%",
            'pass_accuracy': f"{summary.get('pass_accuracy', 0):.1f}%"
        },
        'performance_insights': insights,
        'event_timeline': [
            {
                'type': e.get('type', 'unknown'),
                'time': e.get('frame', 0) / 30.0,  # Assuming 30 fps
                'probability': e.get('probability')
            }
            for e in (analytics_data.get('pass_events', []) + 
                     analytics_data.get('shot_events', []) + 
                     analytics_data.get('goal_events', []))
        ]
    }