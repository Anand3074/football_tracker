from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, jsonify
import os
import uuid
import json
import threading
from werkzeug.utils import secure_filename
from track_football import process_video, generate_match_report
import cv2  # Added

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ANALYTICS_FOLDER = 'analytics'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['ANALYTICS_FOLDER'] = ANALYTICS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.secret_key = 'your_very_secret_key_here'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, ANALYTICS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Store processing status for real-time updates
processing_status = {}

def process_video_async(input_path, output_path, selected_actions, selected_analytics, analytics_path, job_id):
    """Process video in background thread with progress updates"""
    try:
        processing_status[job_id] = {'status': 'processing', 'progress': 10, 'message': 'Starting video analysis with DSS...'}
        
        # Process the video with DSS
        success = process_video(
            input_path, 
            output_path, 
            actions=selected_actions, 
            analytics=selected_analytics, 
            analytics_path=analytics_path
        )
        
        if success:
            processing_status[job_id] = {'status': 'completed', 'progress': 100, 'message': 'Analysis complete! DSS recommendations available.'}
            print(f"✅ Processing completed for job {job_id} with DSS")
        else:
            processing_status[job_id] = {'status': 'error', 'progress': 0, 'message': 'Analysis failed'}
            print(f"❌ Processing failed for job {job_id}")
            
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        processing_status[job_id] = {'status': 'error', 'progress': 0, 'message': error_msg}
        print(f"❌ Processing error for job {job_id}: {error_msg}")
        # Clean up on error
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle old-style form submission (fallback)
        if 'video_file' not in request.files:
            flash('No file provided')
            return redirect(request.url)
        
        file = request.files['video_file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Get form data
            selected_actions = request.form.getlist('actions')
            selected_analytics = request.form.getlist('analytics')
            
            print(f"📋 Selected actions: {selected_actions}")
            print(f"📊 Selected analytics: {selected_analytics}")
            
            # Generate unique ID
            job_id = str(uuid.uuid4())
            
            # Save uploaded file
            original_filename = secure_filename(file.filename)
            input_filename = f"{job_id}_{original_filename}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            file.save(input_path)
            
            # Generate output filenames
            name_without_ext = os.path.splitext(original_filename)[0]
            output_filename = f"processed_{job_id}_{name_without_ext}.mp4"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            
            analytics_filename = f"analytics_{job_id}.json"
            analytics_path = os.path.join(app.config['ANALYTICS_FOLDER'], analytics_filename)
            
            # Start background processing
            thread = threading.Thread(
                target=process_video_async,
                args=(input_path, output_path, selected_actions, selected_analytics, analytics_path, job_id)
            )
            thread.daemon = True
            thread.start()
            
            # Redirect to result page
            return redirect(url_for('show_video', filename=output_filename, analytics=analytics_filename))
        else:
            flash('Invalid file type. Allowed: mp4, avi, mov, mkv')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """AJAX endpoint for file upload"""
    try:
        if 'video_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['video_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, or MKV.'}), 400
        
        # Get form data
        selected_actions = request.form.getlist('actions[]')
        selected_analytics = request.form.getlist('analytics[]')
        
        print(f"📋 Selected actions: {selected_actions}")
        print(f"📊 Selected analytics: {selected_analytics}")
        
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        original_filename = secure_filename(file.filename)
        input_filename = f"{job_id}_{original_filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        print(f"💾 Saved uploaded file: {input_path}")
        
        # Generate output filenames
        name_without_ext = os.path.splitext(original_filename)[0]
        output_filename = f"processed_{job_id}_{name_without_ext}.mp4"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        analytics_filename = f"analytics_{job_id}.json"
        analytics_path = os.path.join(app.config['ANALYTICS_FOLDER'], analytics_filename)
        
        # Start background processing
        thread = threading.Thread(
            target=process_video_async,
            args=(input_path, output_path, selected_actions, selected_analytics, analytics_path, job_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'Video uploaded successfully. Processing started with DSS...',
            'output_filename': output_filename,
            'analytics_filename': analytics_filename
        })
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status for a job"""
    status = processing_status.get(job_id, {'status': 'unknown', 'progress': 0, 'message': 'Job not found'})
    print(f"📡 Status request for {job_id}: {status}")
    return jsonify(status)

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve video files with proper headers for browser playback"""
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    print(f"🎬 Serving video: {filename}")
    print(f"📁 Video path: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return jsonify({'error': 'Video file not found'}), 404
    
    file_size = os.path.getsize(video_path)
    print(f"📏 File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"❌ Video file is empty: {video_path}")
        return jsonify({'error': 'Video file is empty'}), 500
    
    # Simple file extension-based content type detection
    file_ext = filename.lower().split('.')[-1]
    content_types = {
        'mp4': 'video/mp4',
        'avi': 'video/x-msvideo', 
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska',
        'webm': 'video/webm'
    }
    content_type = content_types.get(file_ext, 'video/mp4')
    
    print(f"📹 Using content type: {content_type} for extension: {file_ext}")
    
    try:
        response = send_from_directory(
            app.config['PROCESSED_FOLDER'], 
            filename, 
            as_attachment=False,
            mimetype=content_type
        )
        
        # Set proper headers for video streaming
        response.headers['Content-Type'] = content_type
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Content-Length'] = str(file_size)
        
        print(f"✅ Serving video with headers:")
        print(f"   Content-Type: {content_type}")
        print(f"   Content-Length: {file_size}")
        
        return response
    except Exception as e:
        print(f"❌ Error serving video: {e}")
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500
           
@app.route('/result/<filename>')
def show_video(filename):
    """Display the processed video with analytics and DSS insights"""
    analytics_filename = request.args.get('analytics', '')
    video_url = url_for('serve_video', filename=filename)
    
    print(f"📊 Loading result page for: {filename}")
    print(f"📈 Analytics file: {analytics_filename}")
    
    # Load analytics data if available
    analytics_data = {}
    if analytics_filename:
        analytics_path = os.path.join(app.config['ANALYTICS_FOLDER'], analytics_filename)
        if os.path.exists(analytics_path):
            try:
                with open(analytics_path, 'r') as f:
                    analytics_data = json.load(f)
                print(f"✅ Loaded analytics data with DSS insights")
                print(f"📋 DSS Decision count: {analytics_data.get('summary', {}).get('total_decisions', 0)}")
                if 'dss_insights' in analytics_data:
                    print(f"🧠 DSS Insights available")
            except Exception as e:
                print(f"❌ Error loading analytics: {e}")
                analytics_data = {}
        else:
            print(f"❌ Analytics file not found: {analytics_path}")
    
    # Check if video file exists
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(video_path):
        print(f"❌ Processed video not found: {video_path}")
        flash('Processed video not found. Please try analyzing again.')
        return redirect(url_for('index'))
    
    return render_template('result_dss.html',  # Use the new DSS-enhanced template
                         video_url=video_url, 
                         filename=filename,
                         analytics_data=analytics_data)

@app.route('/analytics/<filename>')
def serve_analytics(filename):
    """Serve analytics data"""
    try:
        return send_from_directory(app.config['ANALYTICS_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({"error": "Analytics not found"}), 404

@app.route('/cleanup', methods=['GET', 'POST'])
def cleanup():
    """Clean up all files"""
    try:
        file_count = 0
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, ANALYTICS_FOLDER]:
            if os.path.exists(folder):
                files = os.listdir(folder)
                for file in files:
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        file_count += 1
        
        # Clear processing status
        processing_status.clear()
        
        print(f"🧹 Cleaned up {file_count} files")
        
        if request.method == 'POST':
            return jsonify({'message': f'All uploaded and processed files ({file_count} files) have been cleaned up'})
        else:
            flash(f'All uploaded and processed files ({file_count} files) have been cleaned up')
            return redirect(url_for('index'))
                
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        if request.method == 'POST':
            return jsonify({'error': f'Error during cleanup: {str(e)}'}), 500
        else:
            flash(f'Error during cleanup: {str(e)}')
            return redirect(url_for('index'))

@app.route('/match_report/<analytics_filename>')
def match_report(analytics_filename):
    """Generate comprehensive match report with DSS insights"""
    try:
        analytics_path = os.path.join(app.config['ANALYTICS_FOLDER'], analytics_filename)
        if os.path.exists(analytics_path):
            with open(analytics_path, 'r') as f:
                analytics_data = json.load(f)
            
            report_data = generate_match_report(analytics_data)
            return render_template('match_report_dss.html', report=report_data)  # Use DSS-enhanced report template
        else:
            return jsonify({'error': 'Analytics data not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting VisionPlay Football Analytics Server with DSS...")
    print(f"📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🎬 Processed folder: {os.path.abspath(PROCESSED_FOLDER)}")
    print(f"📊 Analytics folder: {os.path.abspath(ANALYTICS_FOLDER)}")
    print(f"🧠 Decision Support System: ENABLED")
    app.run(debug=True, host='0.0.0.0', port=5001)