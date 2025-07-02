import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ball_tracking.colormap import colormap_rainbow
from ball_tracking.core import Point2D
from ball_tracking.video_loop import VideoLoop

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Store active tracking sessions
active_sessions: Dict[str, dict] = {}

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class BallTracker:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=128, detectShadows=False)
        self.tracked_positions: List[Point2D] = []
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.resolution = (0, 0)
        
    def initialize(self):
        """Initialize the tracker with video properties"""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.resolution = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        # Initialize background model
        ret, frame = cap.read()
        if ret:
            self.bg_sub.apply(frame, learningRate=1.0)
        
        cap.release()
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame and return tracking results"""
        frame_annotated = frame.copy()
        
        # Filter based on color (green ball detection)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([100, 150, 150]))
        mask_color = cv2.morphologyEx(
            mask_color,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        
        # Filter based on motion
        mask_fg = self.bg_sub.apply(frame, learningRate=0)
        mask_fg = cv2.dilate(
            mask_fg,
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        
        # Combine both masks
        mask = cv2.bitwise_and(mask_color, mask_fg)
        mask = cv2.morphologyEx(
            mask,
            op=cv2.MORPH_OPEN,
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        
        ball_position = None
        confidence = 0.0
        
        # Find largest contour corresponding to the ball
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only track if contour is large enough (filter out noise)
            if area > 100:
                x, y, w, h = cv2.boundingRect(largest_contour)
                center = (x + w // 2, y + h // 2)
                
                # Smooth the trajectory
                if self.tracked_positions:
                    prev_center = self.tracked_positions[-1]
                    alpha = 0.9
                    center = (
                        int((1 - alpha) * prev_center[0] + alpha * center[0]),
                        int((1 - alpha) * prev_center[1] + alpha * center[1]),
                    )
                
                self.tracked_positions.append(center)
                ball_position = center
                confidence = min(1.0, area / 1000.0)  # Normalize confidence
                
                # Draw ball marker
                cv2.circle(frame_annotated, center, 30, (255, 0, 0), 2)
                cv2.circle(frame_annotated, center, 2, (255, 0, 0), 2)
        
        # Draw trajectory
        for i in range(1, len(self.tracked_positions)):
            p1 = self.tracked_positions[i-1]
            p2 = self.tracked_positions[i]
            norm_idx = i / max(1, len(self.tracked_positions))
            color = colormap_rainbow(norm_idx)
            cv2.line(frame_annotated, pt1=p1, pt2=p2, color=color, thickness=2)
        
        # Convert frame to base64 for web display
        _, buffer = cv2.imencode('.jpg', frame_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        frame_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        return {
            'frame_number': self.current_frame,
            'ball_position': ball_position,
            'confidence': confidence,
            'trajectory': self.tracked_positions[-50:] if self.tracked_positions else [],  # Last 50 points
            'frame_image': frame_base64,
            'progress': (self.current_frame / max(1, self.total_frames)) * 100
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = Path(app.config['UPLOAD_FOLDER']) / f"{session_id}_{filename}"
    file.save(str(file_path))
    
    # Initialize tracker
    try:
        tracker = BallTracker(file_path)
        tracker.initialize()
        
        active_sessions[session_id] = {
            'tracker': tracker,
            'file_path': file_path,
            'filename': filename,
            'status': 'ready'
        }
        
        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'total_frames': tracker.total_frames,
            'fps': tracker.fps,
            'resolution': tracker.resolution
        })
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        return jsonify({'error': f'Failed to process video: {str(e)}'}), 500


@app.route('/analyze/<session_id>', methods=['POST'])
def analyze_video(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = active_sessions[session_id]
    tracker = session['tracker']
    
    # Get analysis parameters
    data = request.get_json() or {}
    start_frame = data.get('start_frame', 0)
    end_frame = data.get('end_frame', tracker.total_frames)
    step_size = data.get('step_size', 1)
    
    # Process video frames
    cap = cv2.VideoCapture(str(session['file_path']))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    results = []
    frame_count = start_frame
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracker.current_frame = frame_count
        result = tracker.process_frame(frame)
        results.append(result)
        
        # Skip frames based on step size
        for _ in range(step_size - 1):
            cap.read()
            frame_count += 1
        
        frame_count += 1
    
    cap.release()
    
    # Save results
    session['results'] = results
    session['status'] = 'completed'
    
    return jsonify({
        'session_id': session_id,
        'total_frames_processed': len(results),
        'trajectory': tracker.tracked_positions,
        'analysis_complete': True
    })


@app.route('/session/<session_id>', methods=['GET'])
def get_session_info(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = active_sessions[session_id]
    tracker = session['tracker']
    
    return jsonify({
        'session_id': session_id,
        'filename': session['filename'],
        'status': session['status'],
        'total_frames': tracker.total_frames,
        'fps': tracker.fps,
        'resolution': tracker.resolution,
        'current_frame': tracker.current_frame,
        'trajectory_length': len(tracker.tracked_positions)
    })


@app.route('/results/<session_id>', methods=['GET'])
def get_results(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = active_sessions[session_id]
    if 'results' not in session:
        return jsonify({'error': 'Analysis not completed'}), 400
    
    return jsonify({
        'session_id': session_id,
        'results': session['results'],
        'trajectory': session['tracker'].tracked_positions
    })


@app.route('/cleanup/<session_id>', methods=['DELETE'])
def cleanup_session(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = active_sessions[session_id]
    
    # Remove uploaded file
    if session['file_path'].exists():
        session['file_path'].unlink()
    
    # Remove session
    del active_sessions[session_id]
    
    return jsonify({'message': 'Session cleaned up successfully'})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions)
    })


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main() 