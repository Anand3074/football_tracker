"""
Decision Support System (DSS) for Football Analytics
Provides real-time tactical recommendations based on game state
"""
import cv2
import numpy as np
from collections import defaultdict, deque
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class Action(Enum):
    DRIBBLE = "dribble"
    PASS = "pass"
    SHOOT = "shoot"
    HOLD = "hold"
    CLEAR = "clear"

@dataclass
class TacticalOption:
    action: Action
    confidence: float  # 0-100%
    target_position: Optional[Tuple[float, float]] = None
    target_player: Optional[int] = None
    risk_level: float = 0.0  # 0-100%
    reward_potential: float = 0.0  # 0-100%
    reasoning: List[str] = None

class GameState:
    """Represents current game state for decision making"""
    
    def __init__(self):
        self.score_difference = 0
        self.time_remaining = 0
        self.is_critical_moment = False
        self.phase = "attack"  # attack, defend, transition
        self.team_tactic = "balanced"  # aggressive, defensive, balanced
        
    def update(self, analytics, current_time, game_duration):
        """Update game state based on current situation"""
        self.time_remaining = game_duration - current_time
        
        # Determine if critical moment (last 5 minutes)
        self.is_critical_moment = self.time_remaining < 5 * 60
        
        # Determine phase based on ball position
        if analytics.ball_position:
            x_pos = analytics.ball_position[0]
            field_width = analytics.field_dimensions[1]
            
            if x_pos < field_width * 0.3:
                self.phase = "defend"
            elif x_pos > field_width * 0.7:
                self.phase = "attack"
            else:
                self.phase = "transition"
        
        # Adjust tactics based on score
        if analytics.goals > 0:
            self.score_difference = analytics.goals
            if self.score_difference >= 2:
                self.team_tactic = "defensive"
            elif self.is_critical_moment and self.score_difference == 1:
                self.team_tactic = "defensive"
            else:
                self.team_tactic = "balanced"
        else:
            self.team_tactic = "aggressive"

class SpatialAnalyzer:
    """Analyzes spatial relationships and field zones"""
    
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.height, self.width = frame_shape[:2]
        self.zones = self._create_tactical_zones()
        
    def _create_tactical_zones(self):
        """Create tactical zones for decision making"""
        zones = {
            'defensive': {
                'x_range': (0, self.width * 0.3),
                'priority': 'clearance'
            },
            'midfield': {
                'x_range': (self.width * 0.3, self.width * 0.7),
                'priority': 'possession'
            },
            'attacking': {
                'x_range': (self.width * 0.7, self.width),
                'priority': 'creation'
            },
            'danger_zone': {
                'x_range': (self.width * 0.85, self.width),
                'y_range': (self.height * 0.3, self.height * 0.7),
                'priority': 'shoot'
            }
        }
        return zones
    
    def get_zone(self, position):
        """Get current zone for a position"""
        x, y = position
        
        for zone_name, zone_data in self.zones.items():
            if 'x_range' in zone_data:
                if zone_data['x_range'][0] <= x <= zone_data['x_range'][1]:
                    if 'y_range' not in zone_data or zone_data['y_range'][0] <= y <= zone_data['y_range'][1]:
                        return zone_name
        return 'midfield'
    
    def find_passing_lanes(self, passer_pos, teammates, opponents):
        """Analyze available passing lanes"""
        passing_options = []
        
        for teammate in teammates:
            # Check if direct pass is possible
            has_lane = self._check_passing_lane(passer_pos, teammate['position'], opponents)
            
            if has_lane:
                distance = np.linalg.norm(
                    np.array(passer_pos) - np.array(teammate['position'])
                )
                angle = self._calculate_pass_angle(passer_pos, teammate['position'])
                
                passing_options.append({
                    'player_id': teammate['id'],
                    'position': teammate['position'],
                    'distance': distance,
                    'angle': angle,
                    'risk': self._calculate_pass_risk(teammate, opponents),
                    'reward': self._calculate_pass_reward(teammate['position'])
                })
        
        # Sort by risk/reward ratio
        passing_options.sort(key=lambda x: x['reward'] / (x['risk'] + 1), reverse=True)
        return passing_options
    
    def _check_passing_lane(self, start, end, opponents, threshold=20):
        """Check if pass can be made without interception"""
        for opponent in opponents:
            # Check if opponent is near the line between start and end
            line_vector = np.array(end) - np.array(start)
            point_vector = np.array(opponent['position']) - np.array(start)
            
            line_length = np.linalg.norm(line_vector)
            if line_length == 0:
                continue
                
            projection = np.dot(point_vector, line_vector) / line_length
            
            if 0 <= projection <= line_length:
                projection_point = np.array(start) + (projection / line_length) * line_vector
                distance = np.linalg.norm(np.array(opponent['position']) - projection_point)
                
                if distance < threshold:
                    return False
        return True
    
    def _calculate_pass_angle(self, start, end):
        """Calculate pass angle relative to goal"""
        goal_center = (self.width, self.height / 2)
        pass_vector = np.array(end) - np.array(start)
        goal_vector = np.array(goal_center) - np.array(start)
        
        if np.linalg.norm(pass_vector) == 0 or np.linalg.norm(goal_vector) == 0:
            return 0
            
        cos_angle = np.dot(pass_vector, goal_vector) / (np.linalg.norm(pass_vector) * np.linalg.norm(goal_vector))
        return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
    
    def _calculate_pass_risk(self, teammate, opponents):
        """Calculate risk level for pass"""
        risk = 0
        
        # Check nearby opponents
        for opponent in opponents:
            distance = np.linalg.norm(
                np.array(teammate['position']) - np.array(opponent['position'])
            )
            if distance < 30:
                risk += 30
            elif distance < 60:
                risk += 15
            elif distance < 100:
                risk += 5
        
        return min(100, risk)
    
    def _calculate_pass_reward(self, position):
        """Calculate reward potential for pass to position"""
        x, y = position
        
        # Reward for forward progression
        reward = (x / self.width) * 100
        
        # Bonus for width
        width_factor = 1 - abs(y - self.height/2) / (self.height/2)
        reward += width_factor * 20
        
        return min(100, reward)

class DecisionEngine:
    """Main decision engine for tactical recommendations"""
    
    def __init__(self, frame_shape):
        self.spatial_analyzer = SpatialAnalyzer(frame_shape)
        self.game_state = GameState()
        self.decision_history = deque(maxlen=50)
        self.option_cache = {}
        
        # Weight parameters for decision making
        self.weights = {
            'shoot': {
                'position': 0.35,
                'angle': 0.25,
                'pressure': 0.20,
                'game_state': 0.15,
                'confidence': 0.05
            },
            'pass': {
                'safety': 0.30,
                'progression': 0.30,
                'opportunity': 0.25,
                'game_state': 0.15
            },
            'dribble': {
                'space': 0.35,
                'support': 0.25,
                'pressure': 0.25,
                'game_state': 0.15
            }
        }
        
    def analyze_situation(self, possessor_data, teammates, opponents, 
                          ball_data, analytics, current_time):
        """Complete analysis of current situation"""
        
        # Update game state
        self.game_state.update(analytics, current_time, 600)  # 10 minutes
        
        # Generate tactical options
        options = []
        
        # 1. Shoot option
        shoot_option = self.evaluate_shoot_option(
            possessor_data, ball_data, opponents, analytics
        )
        if shoot_option:
            options.append(shoot_option)
        
        # 2. Pass options
        pass_options = self.evaluate_pass_options(
            possessor_data, teammates, opponents, ball_data
        )
        options.extend(pass_options)
        
        # 3. Dribble option
        dribble_option = self.evaluate_dribble_option(
            possessor_data, opponents, analytics
        )
        if dribble_option:
            options.append(dribble_option)
        
        # Sort by confidence
        options.sort(key=lambda x: x.confidence, reverse=True)
        
        return options[:3]  # Return top 3 options
    
    def evaluate_shoot_option(self, possessor, ball_data, opponents, analytics):
        """Evaluate shooting opportunity"""
        
        if not possessor or not ball_data:
            return None
            
        ball_pos = ball_data['position']
        current_zone = self.spatial_analyzer.get_zone(ball_pos)
        
        # Only consider shoot if in attacking zone
        if current_zone not in ['attacking', 'danger_zone']:
            return None
            
        # Calculate shooting confidence
        position_score = self._evaluate_shoot_position(ball_pos)
        angle_score = self._evaluate_shoot_angle(ball_pos)
        pressure_score = self._evaluate_pressure(ball_pos, opponents)
        game_state_score = self._evaluate_game_state_for_shoot()
        
        # Weighted confidence
        confidence = (
            position_score * self.weights['shoot']['position'] +
            angle_score * self.weights['shoot']['angle'] +
            (100 - pressure_score) * self.weights['shoot']['pressure'] +
            game_state_score * self.weights['shoot']['game_state']
        )
        
        reasoning = []
        if confidence > 70:
            reasoning.append("Excellent shooting position")
        elif confidence > 50:
            reasoning.append("Decent shooting opportunity")
        else:
            reasoning.append("Low percentage shot")
            
        if pressure_score > 50:
            reasoning.append("Under pressure from defenders")
        if angle_score > 70:
            reasoning.append("Good angle to goal")
            
        return TacticalOption(
            action=Action.SHOOT,
            confidence=confidence,
            risk_level=pressure_score,
            reward_potential=position_score * 0.7 + angle_score * 0.3,
            reasoning=reasoning
        )
    
    def evaluate_pass_options(self, possessor, teammates, opponents, ball_data):
        """Evaluate all possible pass options"""
        options = []
        
        if not possessor or not ball_data:
            return options
            
        possessor_pos = possessor['position']
        
        # Find available passing lanes
        passing_options = self.spatial_analyzer.find_passing_lanes(
            possessor_pos, teammates, opponents
        )
        
        for pass_option in passing_options[:3]:  # Top 3 passing options
            # Calculate confidence for this pass
            safety_score = 100 - pass_option['risk']
            progression_score = self._evaluate_pass_progression(pass_option['position'])
            opportunity_score = self._evaluate_pass_opportunity(pass_option)
            game_state_score = self._evaluate_game_state_for_pass()
            
            confidence = (
                safety_score * self.weights['pass']['safety'] +
                progression_score * self.weights['pass']['progression'] +
                opportunity_score * self.weights['pass']['opportunity'] +
                game_state_score * self.weights['pass']['game_state']
            )
            
            reasoning = []
            if safety_score > 70:
                reasoning.append("Safe pass option")
            if progression_score > 70:
                reasoning.append("Progresses play forward")
            if pass_option['distance'] < 50:
                reasoning.append("Short, controlled pass")
            elif pass_option['distance'] > 100:
                reasoning.append("Long range pass")
                
            options.append(TacticalOption(
                action=Action.PASS,
                confidence=confidence,
                target_player=pass_option['player_id'],
                target_position=pass_option['position'],
                risk_level=pass_option['risk'],
                reward_potential=pass_option['reward'],
                reasoning=reasoning
            ))
        
        return options
    
    def evaluate_dribble_option(self, possessor, opponents, analytics):
        """Evaluate dribbling opportunity"""
        
        if not possessor:
            return None
            
        possessor_pos = possessor['position']
        current_zone = self.spatial_analyzer.get_zone(possessor_pos)
        
        # Calculate space available
        space_score = self._evaluate_dribbling_space(possessor_pos, opponents)
        
        # Calculate support from teammates
        # (Simplified - would need teammate positions)
        support_score = 50  # Default value
        
        # Calculate pressure from defenders
        pressure_score = self._evaluate_pressure(possessor_pos, opponents)
        
        # Game state influence
        game_state_score = self._evaluate_game_state_for_dribble()
        
        confidence = (
            space_score * self.weights['dribble']['space'] +
            support_score * self.weights['dribble']['support'] +
            (100 - pressure_score) * self.weights['dribble']['pressure'] +
            game_state_score * self.weights['dribble']['game_state']
        )
        
        reasoning = []
        if space_score > 70:
            reasoning.append("Plenty of space to run into")
        if pressure_score > 70:
            reasoning.append("Under heavy pressure")
        if current_zone == 'defensive':
            reasoning.append("Dribbling out of defensive zone")
        elif current_zone == 'attacking':
            reasoning.append("Attacking with purpose")
            
        return TacticalOption(
            action=Action.DRIBBLE,
            confidence=confidence,
            risk_level=pressure_score,
            reward_potential=space_score * 0.6 + (100 - pressure_score) * 0.4,
            reasoning=reasoning
        )
    
    def _evaluate_shoot_position(self, position):
        """Evaluate quality of shooting position"""
        x, y = position
        h, w = self.frame_shape[:2] if hasattr(self, 'frame_shape') else (720, 1280)
        
        # Distance to goal
        distance_to_goal = np.linalg.norm(np.array(position) - np.array([w, h/2]))
        max_distance = np.linalg.norm([w, h/2])
        distance_score = 100 * (1 - distance_to_goal / max_distance)
        
        # Central position bonus
        central_bonus = 100 * (1 - abs(y - h/2) / (h/2))
        
        return min(100, distance_score * 0.6 + central_bonus * 0.4)
    
    def _evaluate_shoot_angle(self, position):
        """Evaluate shooting angle"""
        x, y = position
        h, w = self.frame_shape[:2] if hasattr(self, 'frame_shape') else (720, 1280)
        
        goal_center = (w, h/2)
        dx = goal_center[0] - x
        dy = goal_center[1] - y
        
        if dx == 0:
            return 0
            
        angle = abs(math.degrees(math.atan2(dy, dx)))
        
        # Smaller angle = better shooting opportunity
        angle_score = max(0, 100 - (angle * 2))
        return angle_score
    
    def _evaluate_pressure(self, position, opponents):
        """Calculate pressure from nearby opponents"""
        if not opponents:
            return 0
            
        pressure = 0
        for opponent in opponents:
            distance = np.linalg.norm(
                np.array(position) - np.array(opponent['position'])
            )
            
            if distance < 30:
                pressure += 35
            elif distance < 60:
                pressure += 20
            elif distance < 100:
                pressure += 10
            elif distance < 150:
                pressure += 5
                
        return min(100, pressure)
    
    def _evaluate_pass_progression(self, target_position):
        """Evaluate how much pass progresses play forward"""
        x, _ = target_position
        w = self.frame_shape[1] if hasattr(self, 'frame_shape') else 1280
        
        progression = (x / w) * 100
        return progression
    
    def _evaluate_pass_opportunity(self, pass_option):
        """Evaluate the opportunity created by pass"""
        reward = pass_option['reward']
        
        # Bonus for passing to player in better position
        x = pass_option['position'][0]
        w = self.frame_shape[1] if hasattr(self, 'frame_shape') else 1280
        
        if x > w * 0.7:  # Attacking zone
            reward += 20
            
        return min(100, reward)
    
    def _evaluate_dribbling_space(self, position, opponents):
        """Evaluate space available for dribbling"""
        h, w = self.frame_shape[:2] if hasattr(self, 'frame_shape') else (720, 1280)
        x, y = position
        
        # Space to goal
        space_to_goal = w - x
        
        # Check opponent positions
        occupied_spaces = []
        for opponent in opponents:
            opp_x, opp_y = opponent['position']
            occupied_spaces.append((opp_x, opp_y))
        
        # Find largest gap
        if occupied_spaces:
            sorted_x = sorted([opp_x for opp_x, _ in occupied_spaces])
            gaps = []
            current_pos = x
            
            for opp_x in sorted_x:
                if opp_x > current_pos:
                    gaps.append(opp_x - current_pos)
                    current_pos = opp_x
            
            max_gap = max(gaps) if gaps else space_to_goal
        else:
            max_gap = space_to_goal
            
        space_score = min(100, (max_gap / w) * 100)
        return space_score
    
    def _evaluate_game_state_for_shoot(self):
        """Evaluate game state influence on shooting decision"""
        score = 50  # Baseline
        
        if self.game_state.is_critical_moment:
            if self.game_state.score_difference <= 0:
                score += 30  # Need goal, take risks
            else:
                score -= 20  # Protect lead
                
        if self.game_state.team_tactic == "aggressive":
            score += 20
        elif self.game_state.team_tactic == "defensive":
            score -= 15
            
        return min(100, max(0, score))
    
    def _evaluate_game_state_for_pass(self):
        """Evaluate game state influence on passing decision"""
        score = 60  # Baseline - passing is generally good
        
        if self.game_state.is_critical_moment:
            if self.game_state.score_difference <= 0:
                score += 10  # Need to create chances
            else:
                score += 20  # Keep possession, run down clock
                
        return min(100, max(0, score))
    
    def _evaluate_game_state_for_dribble(self):
        """Evaluate game state influence on dribbling decision"""
        score = 40  # Baseline - dribbling has risk
        
        if self.game_state.is_critical_moment:
            if self.game_state.score_difference <= 0:
                score += 25  # Need to create something
            else:
                score -= 15  # Avoid losing possession
                
        if self.game_state.phase == "attack":
            score += 20
        elif self.game_state.phase == "defend":
            score -= 15
            
        return min(100, max(0, score))

class DecisionVisualizer:
    """Visualizes tactical decisions on frame"""
    
    def __init__(self):
        self.colors = {
            Action.SHOOT: (100, 100, 255),    # Red
            Action.PASS: (255, 255, 100),      # Yellow
            Action.DRIBBLE: (100, 255, 100),   # Green
            Action.HOLD: (255, 100, 255),      # Purple
            Action.CLEAR: (100, 255, 255)      # Cyan
        }
        
    def draw_decision_overlay(self, frame, options, possessor_pos):
        """Draw tactical decision overlay"""
        if not options:
            return frame
            
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for decision panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 320, h - 220), (w - 20, h - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "⚽ TACTICAL DECISIONS", (w - 300, h - 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw options
        y_offset = h - 160
        for i, option in enumerate(options[:3]):
            color = self.colors.get(option.action, (200, 200, 200))
            
            # Action name and confidence
            action_text = f"{i+1}. {option.action.value.upper()}"
            cv2.putText(frame, action_text, (w - 300, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            
            # Confidence bar
            bar_x = w - 220
            bar_y = y_offset - 12
            bar_width = 150
            bar_height = 8
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (80, 80, 80), -1)
            
            # Fill
            fill_width = int(bar_width * (option.confidence / 100))
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         color, -1)
            
            # Percentage
            cv2.putText(frame, f"{option.confidence:.0f}%",
                       (w - 60, y_offset - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Reasoning (first line)
            if option.reasoning:
                cv2.putText(frame, option.reasoning[0][:25],
                           (w - 300, y_offset + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 55
        
        return frame
    
    def draw_passing_lanes(self, frame, possessor_pos, pass_options):
        """Draw passing lanes visualization"""
        for option in pass_options[:2]:  # Show top 2 passing options
            if option.target_position:
                # Draw line
                cv2.line(frame, 
                        (int(possessor_pos[0]), int(possessor_pos[1])),
                        (int(option.target_position[0]), int(option.target_position[1])),
                        self.colors[Action.PASS], 2)
                
                # Draw confidence indicator
                mid_point = (
                    int((possessor_pos[0] + option.target_position[0]) / 2),
                    int((possessor_pos[1] + option.target_position[1]) / 2)
                )
                
                # Risk/reward indicator
                risk_reward = option.reward_potential - option.risk_level
                color = (0, 255, 0) if risk_reward > 0 else (0, 255, 255)
                
                cv2.circle(frame, mid_point, 5, color, -1)
        
        return frame
    
    def draw_shoot_indicator(self, frame, ball_pos, probability):
        """Draw shooting indicator"""
        if probability < 30:
            return frame
            
        x, y = int(ball_pos[0]), int(ball_pos[1])
        
        # Draw trajectory to goal
        h, w = frame.shape[:2]
        goal_pos = (w, int(h/2))
        
        # Draw shooting arc
        cv2.line(frame, (x, y), goal_pos, self.colors[Action.SHOOT], 2)
        
        # Draw probability circle
        radius = 30 + int(probability / 2)
        cv2.circle(frame, (x, y), radius, self.colors[Action.SHOOT], 2)
        cv2.circle(frame, (x, y), radius - 5, self.colors[Action.SHOOT], 1)
        
        # Text
        cv2.putText(frame, f"SHOT {probability:.0f}%", (x - 30, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[Action.SHOOT], 2)
        
        return frame