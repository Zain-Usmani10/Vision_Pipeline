# Monocular Spatial Vision Pipeline

This repository contains a high performance monocular vision pipeline engineered for autonomous robotics and spatial reconstruction. The system transforms flat 2D video feeds into mathematically accurate 3D point clouds without relying on physical LiDAR sensors or stereo camera hardware.

## 1) Algorithmic Foundations

### 1. Camera Intrinsic Unprojection and Focal Length
The most critical variable in monocular spatial vision is the internal geometry of the camera lens. We must convert digital pixels back into physical light rays. To do this we calculate the focal length in pixels using the 35mm equivalent focal length provided by standard hardware specifications.

The pipeline employs the unprojection formula
$$f_{px} = \frac{f_{35} \times w}{36}$$

The variable $w$ represents the horizontal pixel resolution of the video frame. The anchor value of 36 represents the physical width in millimeters of a standard 35mm film frame. By mapping the equivalent focal length to the digital resolution we create a universal geometric baseline for any smartphone or web camera.

Electronic Image Stabilization often sabotages this math by inducing an invisible software crop on the sensor. This crop artificially narrows the field of view and increases the true focal length. Our pipeline allows for dynamic overriding of the $f_{px}$ value based on physical checkerboard calibration scripts to guarantee sub-millimeter precision during the 3D unprojection phase.

### 2. Dense Depth Inference
We utilize the Depth Anything 3 architecture to extract depth from flat images. Traditional stereo vision relies on calculating pixel disparity between two distinct lenses. This pipeline uses a single stream transformer network that understands semantic context and physical scale.

The model analyzes texture gradients and structural edges to assign an absolute metric distance to every single pixel. The output is a high resolution depth map where the Z axis is strictly defined in physical meters rather than arbitrary relative values. This metric scale is mandatory for autonomous navigation algorithms that require real world distance calculations to prevent collisions.

### 3. Temporal Pose Estimation
To map a room we must calculate the exact spatial trajectory of the camera over time. We extract visual feature points across sequential video frames using optical flow techniques. 

By tracking how these points move across the 2D plane we can calculate the Essential Matrix. We apply a RANSAC algorithm to filter out dynamic objects like moving humans or vehicles. This isolates the static background and allows us to mathematically derive the translation and rotation vectors of the camera lens for every millisecond of footage.

### 4. Pose Graph Optimization
Dead reckoning always introduces drift. As the camera tracking algorithm processes thousands of frames tiny mathematical errors compound and cause the 3D map to warp. 

We deploy a Pose Graph Optimization back end to eliminate this drift. When the camera observes a landmark it has seen before the algorithm registers a loop closure. It then solves a massive non linear least squares optimization problem to distribute the geometric error backward through the entire trajectory. This forces the physical map to stay rigid and perfectly aligned with reality.

### 5. Point Cloud Generation
The final stage merges the metric depth map with the optimized camera trajectory. We project each 2D pixel into 3D Cartesian coordinates using the depth value $Z$ and the optical center coordinates $c_x$ and $c_y$

$$X = \frac{(u - c_x) \times Z}{f_x}$$
$$Y = \frac{(v - c_y) \times Z}{f_y}$$

We apply statistical outlier removal to eliminate floating noise artifacts. The resulting data structure is a dense and metric accurate point cloud ready for robotic path planning or spatial visualization frameworks.

## 2) Directory Blueprint

You must adhere to this exact folder structure for the pipeline to execute correctly. The model weights must be placed exactly as shown below.

```text
vision_pipeline/
├── DA3_Metric_Model/
│   ├── README.md
│   ├── config.json          <-- (Must be downloaded manually)
│   └── model.safetensors    <-- (Must be downloaded manually)
├── scripts/                 <-- (Core spatial math and processing)
├── static/                  <-- (Frontend web viewer and JS)
├── Depth-Anything-3/        <-- (Flattened framework API files)
├── app.py                   <-- (Flask web server entry point)
├── main.py                  <-- (Local backend execution loop)
├── requirements.txt
└── .gitignore
```

## 3) Frontend Architecture and Capabilities

The pipeline includes a custom built web application interface engineered to handle heavy 3D rendering and user management without crashing the browser thread.

* **Hardware Preset Catalog** The interface features a database of over 100 mobile devices. Users simply select their phone model and the frontend automatically injects the correct 35mm equivalent focal length into the backend python engine.
* **Asynchronous Processing** Video processing blocks the main server thread. The frontend utilizes asynchronous polling to track the Python backend progress in real time and updates a dynamic progress bar for each stage of the extraction and optimization loop.
* **Client Side 3D Rendering** The viewer utilizes Three.js to render massive PLY point clouds directly in the browser using WebGL. This offloads the graphical rendering burden from the server directly to the local GPU of the user.
* **Data Management** Users can visually manage their completed pipeline runs. The interface allows for instant deletion of heavy point cloud data or immediate transition into the interactive 3D visualizer tool.

## 4) Deployment and Execution

This pipeline supports both local validation testing and production grade remote server deployment.

### A. Local CPU Deployment
You can run this entire pipeline on a local machine for testing and validation. Compute times will be significantly slower than a dedicated GPU server but the mathematical output remains absolutely identical.

**Step 1 Clone the repository**
```bash
git clone git@github.com:Zain-Usmani10/Vision_Pipeline.git
cd Vision_Pipeline
```

**Step 2 Install requirements**
Ensure you have Python 3.10 or higher installed. Create and activate a virtual environment before installing the dependencies.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 3 Download inference weights**
The Depth Anything 3 model weights are too massive for standard version control storage. You must acquire them manually.
1. Navigate to the official Hugging Face repository for Depth Anything 3.
2. Download the `model.safetensors` and `config.json` files.
3. Place both files directly inside the `DA3_Metric_Model/` directory.

**Step 4 Execute the backend**
Launch the python application to begin processing your video files locally.
```bash
python app.py
```

### B. Remote GPU Server Deployment
If you are deploying this pipeline on a remote headless Linux server you must use a production WSGI server and a secure networking tunnel to expose the web interface.

**Step 1 Provision the production environment**
Follow Steps 1 through 3 from the Local CPU Deployment guide to clone the code and download the required weights.

**Step 2 Install the Ngrok Agent**
Download and install the Ngrok tunneling tool to securely route traffic from your remote server to the internet without opening firewall ports.
```bash
curl -sSL [https://ngrok-agent.s3.amazonaws.com/ngrok.asc](https://ngrok-agent.s3.amazonaws.com/ngrok.asc) | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb [https://ngrok-agent.s3.amazonaws.com](https://ngrok-agent.s3.amazonaws.com) buster main" | tee /etc/apt/sources.list.d/ngrok.list
apt update && apt install ngrok
```

**Step 3 Authenticate Ngrok**
You must create a free account at ngrok.com to obtain your authentication token and static domain name.
```bash
ngrok config add-authtoken <YOUR_TOKEN_HERE>
```

**Step 4 Launch the Gunicorn Server**
Do not use the standard python command for a production environment. Bind the application to a local port using Gunicorn and run it as a background daemon process.
```bash
nohup gunicorn -w 1 -b 127.0.0.1:5000 app:app > /root/vision_pipeline/server.log 2>&1 &
```

**Step 5 Initialize the Secure Tunnel**
Route the local Gunicorn port through the Ngrok daemon to generate your static public HTTPS URL. Replace the domain placeholder with your assigned free domain.
```bash
nohup ngrok http --url=<YOUR_STATIC_DOMAIN_HERE> 127.0.0.1:5000 > /root/vision_pipeline/ngrok.log 2>&1 &
```

**Step 6 Process Management and Updates**
When you need to restart the server or push new code updates you must terminate the existing background processes to free up the network port before restarting Gunicorn.
```bash
pkill gunicorn
<<<<<<< HEAD
```
=======
```
>>>>>>> 2cd68d8 (Finalized v1)
