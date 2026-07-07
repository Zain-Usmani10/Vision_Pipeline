import os
import threading
import queue
import traceback
import numpy as np
import glob
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from scripts.stream_gatekeeper import StreamGatekeeper
from scripts.depth_inference import DA3DepthWorker
from scripts.sliding_window_slam import SlidingWindowSLAM
from scripts.memory_manager import SubMapManager

app = Flask(__name__, template_folder='/root/vision_pipeline/templates')
UPLOAD_FOLDER = "/root/vision_pipeline/uploads"
OUTPUT_FOLDER = "/root/vision_pipeline/outputs/active_sessions"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pipeline_status = {"status": "idle", "frame_idx": 0, "point_count": 0, "completed": False}

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def wrap_worker(worker_func, name):
    def wrapped():
        try:
            print(f"[{name}] Thread initialized and running")
            worker_func()
            print(f"[{name}] Thread completed successfully")
        except Exception as e:
            print(f"[{name}] FATAL CRASH {e}")
            traceback.print_exc()
    return wrapped

def run_pipeline_async(video_path, focal_length, width, height):
    global pipeline_status
    pipeline_status["status"] = "processing"
    pipeline_status["completed"] = False
    pipeline_status["frame_idx"] = 0
    pipeline_status["point_count"] = 0
    
    engine_path = "/root/vision_pipeline/DA3_Metric_Model/da3_metric_518_fp16.engine"
    
    frame_queue = queue.Queue(maxsize=30)
    slam_queue = queue.Queue(maxsize=30)
    memory_queue = queue.Queue(maxsize=30)
    
    gatekeeper = StreamGatekeeper(video_path, frame_queue, skip_rate=1)
    depth_worker = DA3DepthWorker(engine_path, frame_queue, slam_queue, focal_length)
    slam_worker = SlidingWindowSLAM(slam_queue, memory_queue, focal_length, width, height)
    memory_worker = SubMapManager(memory_queue, focal_length, width, height, pipeline_status)
    
    t1 = threading.Thread(target=wrap_worker(gatekeeper.stream_loop, "Gatekeeper"))
    t2 = threading.Thread(target=wrap_worker(depth_worker.run, "DepthInference"))
    t3 = threading.Thread(target=wrap_worker(slam_worker.run, "SLAMWorker"))
    t4 = threading.Thread(target=wrap_worker(memory_worker.run, "MemoryManager"))
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    
    t1.join()
    frame_queue.put(None)
    t2.join()
    slam_queue.put(None)
    t3.join()
    memory_queue.put(None)
    t4.join()
    
    pipeline_status["status"] = "finished"
    pipeline_status["completed"] = True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
        
    file = request.files["video"]
    
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    if file:
        old_uploads = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*"))
        for f in old_uploads:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to delete upload {f}")

        old_outputs = glob.glob(os.path.join(OUTPUT_FOLDER, "*"))
        for f in old_outputs:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to delete output {f}")

        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(saved_path)
        
        width = 1920
        height = 1080
        focal_length_px = 1623.0
        
        threading.Thread(target=run_pipeline_async, args=(saved_path, focal_length_px, width, height)).start()
        
        return redirect(url_for("viewer"))

@app.route("/viewer")
def viewer():
    return render_template("viewer.html")

@app.route("/api/status")
def get_status():
    return jsonify(pipeline_status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)