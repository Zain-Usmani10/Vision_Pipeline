import os
import json
import uuid
import threading
import sys
import time
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER    = '/root/vision_pipeline/uploads'
OUTPUTS_FOLDER   = '/root/vision_pipeline/outputs'
MAX_VIDEOS       = 10
MAX_FILE_SIZE_MB = 500
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

jobs = {} 

def get_all_videos():
    videos = []
    if not os.path.exists(UPLOAD_FOLDER): return []
    for f in os.listdir(UPLOAD_FOLDER):
        if f.endswith('_metadata.json'):
            path = os.path.join(UPLOAD_FOLDER, f)
            with open(path) as fp:
                try:
                    meta = json.load(fp)
                    videos.append(meta)
                except: continue
    videos.sort(key=lambda x: x.get('uploaded_at', ''))
    return videos

def enforce_video_limit():
    videos = get_all_videos()
    while len(videos) >= MAX_VIDEOS:
        oldest = videos.pop(0)
        video_name = oldest['video_name']
        video_path = os.path.join(UPLOAD_FOLDER, f"{video_name}.mp4")
        if os.path.exists(video_path): os.remove(video_path)
        meta_path = os.path.join(UPLOAD_FOLDER, f"{video_name}_metadata.json")
        if os.path.exists(meta_path): os.remove(meta_path)
        output_path = os.path.join(OUTPUTS_FOLDER, video_name)
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)

def run_pipeline(job_id, video_path, video_name, focal_35mm):
    last_time = time.time()
    last_stage = None

    def update(stage, frame=None, total=None):
        nonlocal last_time, last_stage
        now = time.time()
        if last_stage: jobs[job_id]['step_times'][last_stage] = round(now - last_time, 1)
        last_time = now
        last_stage = stage
        jobs[job_id].update({'status': 'processing', 'stage': stage, 'frame': frame, 'total_frames': total})
        status_path = os.path.join(OUTPUTS_FOLDER, video_name, 'status.json')
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        with open(status_path, 'w') as f: json.dump(jobs[job_id], f)

    try:
        sys.path.append('/root/vision_pipeline/scripts')
        from frame_extraction import extract_frames
        from vid_depth import run_depth
        from pose_estimation import run_pose_estimation
        from point_cloud import run_point_cloud
        from merge_pointclouds import merge_pointclouds
        from generate_pose_graph import generate_pose_graph

        update('Extracting frames')
        meta = extract_frames(video_path)

        if focal_35mm:
            try:
                f_35 = float(focal_35mm)
                w = meta["width"]
                h = meta["height"]
                fx = (f_35 * max(w, h)) / 36.0
                meta["intrinsics"]["fx"] = fx
                meta["intrinsics"]["fy"] = fx
                meta_path = f"/root/vision_pipeline/uploads/{video_name}_metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print(f"Failed to patch intrinsics: {e}")

        update('Running depth estimation', frame=0, total=meta['frame_count'])
        run_depth(video_name)
        update('Running pose estimation', frame=0, total=meta['frame_count'])
        run_pose_estimation(video_name)
        update('Generating point clouds', frame=0, total=meta['frame_count'])
        run_point_cloud(video_name)
        update('Merging point clouds')
        merge_pointclouds(video_name)
        update('Generating pose graph')
        generate_pose_graph(video_name)
        
        jobs[job_id]['status'] = 'done'
        status_path = os.path.join(OUTPUTS_FOLDER, video_name, 'status.json')
        with open(status_path, 'w') as f: json.dump(jobs[job_id], f)
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        status_path = os.path.join(OUTPUTS_FOLDER, video_name, 'status.json')
        with open(status_path, 'w') as f: json.dump(jobs[job_id], f)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/viewer')
def viewer():
    return send_from_directory('static', 'viewer.html')

@app.route('/videos', methods=['GET'])
def list_videos():
    return jsonify(get_all_videos())

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files: return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    focal_35mm = request.form.get('focal_35mm', '24')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = f"video_{timestamp}"
    video_path = os.path.join(UPLOAD_FOLDER, f"{video_name}.mp4")
    enforce_video_limit()
    file.save(video_path)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'job_id': job_id, 'video_name': video_name, 'status': 'queued', 'step_times': {}}
    thread = threading.Thread(target=run_pipeline, args=(job_id, video_path, video_name, focal_35mm))
    thread.daemon = True
    thread.start()
    return jsonify({'job_id': job_id, 'video_name': video_name})

@app.route('/delete/<video_name>', methods=['DELETE'])
def delete_video(video_name):
    video_path = os.path.join(UPLOAD_FOLDER, f"{video_name}.mp4")
    meta_path = os.path.join(UPLOAD_FOLDER, f"{video_name}_metadata.json")
    output_path = os.path.join(OUTPUTS_FOLDER, video_name)
    
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
        
    return jsonify({'success': True})

@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in jobs: return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])

@app.route('/outputs/<video_name>/<path:filepath>')
def serve_output(video_name, filepath):
    directory = os.path.join(OUTPUTS_FOLDER, video_name)
    return send_from_directory(directory, filepath)

@app.route('/depth_frames/<video_name>')
def depth_frames(video_name):
    viz_dir = os.path.join(OUTPUTS_FOLDER, video_name, 'visualizations')
    if not os.path.exists(viz_dir):
        return jsonify([])
    files = sorted([f for f in os.listdir(viz_dir) if f.startswith('depth_') and f.endswith('.png')])
    return jsonify(files)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)