"""
Chain Fit Studio - Web App with Flask
Run this and open http://localhost:5000 in your browser
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import os
import math
from collections import deque
import json

app = Flask(__name__)

# ================== CONFIG ==================
CHAIN_FOLDER = "chains"
os.makedirs(CHAIN_FOLDER, exist_ok=True)

# Load chains
chains = []
chain_names = []
print("Loading chains from 'chains/'...")
for file in sorted(os.listdir(CHAIN_FOLDER)):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(CHAIN_FOLDER, file)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.shape[2] == 3:
                h, w = img.shape[:2]
                alpha = np.full((h, w, 1), 255, np.uint8)
                img = np.dstack([img, alpha])
            chains.append(img)
            chain_names.append(os.path.splitext(file)[0])
            print(f"   Check: {file}")

if not chains:
    raise FileNotFoundError("Error: Put chain PNGs in 'chains/' folder!")

# ================== GLOBAL STATE ==================
class ChainFitState:
    def __init__(self):
        self.current_idx = 0
        self.BASE_WIDTH = 300
        self.chain_img = None
        self.chain_h = 0
        self.chain_w = 0
        self.original_chain = None
        
        # Settings
        self.show_neck_dot = True
        self.brightness_auto = True
        self.chain_scale = 1.0
        self.vertical_offset = 0.20
        
        # Mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmarks
        self.JAW_LEFT = 234
        self.JAW_RIGHT = 454
        self.CHIN = 152
        self.NOSE_TIP = 4
        
        # Buffers
        self.neck_buffer = deque(maxlen=8)
        self.scale_buffer = deque(maxlen=6)
        self.tilt_buffer = deque(maxlen=5)
        
        self.frame_count = 0
        self.prev_depth = 500
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.load_chain()
    
    def load_chain(self):
        img = chains[self.current_idx]
        h, w = img.shape[:2]
        scale = self.BASE_WIDTH / w
        self.chain_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        self.original_chain = self.chain_img.copy()
        self.chain_h, self.chain_w = self.chain_img.shape[:2]
    
    def get_accurate_neck_position(self, landmarks, img_w, img_h):
        jaw_l = np.array([landmarks[self.JAW_LEFT].x * img_w, landmarks[self.JAW_LEFT].y * img_h])
        jaw_r = np.array([landmarks[self.JAW_RIGHT].x * img_w, landmarks[self.JAW_RIGHT].y * img_h])
        chin = np.array([landmarks[self.CHIN].x * img_w, landmarks[self.CHIN].y * img_h])
        nose = np.array([landmarks[self.NOSE_TIP].x * img_w, landmarks[self.NOSE_TIP].y * img_h])

        jaw_mid = (jaw_l + jaw_r) / 2
        face_length = np.linalg.norm(nose - chin)
        jaw_width = np.linalg.norm(jaw_r - jaw_l)

        vertical_offset = face_length * 0.3
        width_factor = min(jaw_width / img_w, 0.3)
        vertical_offset += width_factor * face_length * 0.8

        neck_x = jaw_mid[0]
        neck_y = chin[1] + vertical_offset

        neck_pos = np.array([neck_x, neck_y])
        self.neck_buffer.append(neck_pos)
        return np.mean(self.neck_buffer, axis=0).astype(int)
    
    def get_head_tilt(self, landmarks, img_w, img_h):
        try:
            image_points = np.float32([
                [landmarks[self.NOSE_TIP].x * img_w, landmarks[self.NOSE_TIP].y * img_h],
                [landmarks[self.CHIN].x * img_w, landmarks[self.CHIN].y * img_h],
                [landmarks[self.JAW_LEFT].x * img_w, landmarks[self.JAW_LEFT].y * img_h],
                [landmarks[self.JAW_RIGHT].x * img_w, landmarks[self.JAW_RIGHT].y * img_h],
            ])
            model_points = np.float32([[0,0,0], [0,-70,-50], [-80,-30,-50], [80,-30,-50]])
            focal = img_w * 0.8
            center = (img_w/2, img_h/2)
            camera_matrix = np.array([[focal,0,center[0]],[0,focal,center[1]],[0,0,1]], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, None,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
            if not success: return 0, 500

            rot_mat, _ = cv2.Rodrigues(rvec)
            yaw = math.degrees(math.asin(-rot_mat[0,2]))
            self.tilt_buffer.append(yaw)
            return np.mean(self.tilt_buffer), tvec[2][0]
        except:
            return 0, 500
    
    def calculate_optimal_scale(self, landmarks, img_w, img_h, chain_w, depth):
        jaw_l = np.array([landmarks[self.JAW_LEFT].x * img_w, landmarks[self.JAW_LEFT].y * img_h])
        jaw_r = np.array([landmarks[self.JAW_RIGHT].x * img_w, landmarks[self.JAW_RIGHT].y * img_h])
        jaw_width = np.linalg.norm(jaw_r - jaw_l)

        face_ratio = jaw_width / img_w
        target_w = jaw_width * (1.4 + face_ratio * 0.8)
        scale = target_w / chain_w

        ref_depth = 500
        depth_factor = ref_depth / max(depth, 50)
        scale *= np.clip(depth_factor * 0.95, 0.8, 1.2)

        scale = np.clip(scale, 0.5, 3.5)
        scale *= self.chain_scale
        self.scale_buffer.append(scale)
        return np.mean(self.scale_buffer)
    
    def overlay_chain(self, bg, chain, x, y):
        h, w = chain.shape[:2]
        if h <= 0 or w <= 0: return bg

        y1, y2 = max(0, y), min(bg.shape[0], y + h)
        x1, x2 = max(0, x), min(bg.shape[1], x + w)
        if y1 >= y2 or x1 >= x2: return bg

        chain_crop = chain[y1-y:y2-y, x1-x:x2-x, :]
        bg_crop = bg[y1:y2, x1:x2].copy()
        alpha = chain_crop[:, :, 3:4] / 255.0

        for c in range(3):
            bg_crop[:, :, c] = (1 - alpha[:,:,0]) * bg_crop[:,:,c] + alpha[:,:,0] * chain_crop[:,:,c]

        bg[y1:y2, x1:x2] = bg_crop
        return bg

state = ChainFitState()

# ================== ROUTES ==================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chains')
def get_chains():
    return jsonify({'chains': chain_names, 'current': state.current_idx})

@app.route('/api/chain/select', methods=['POST'])
def select_chain():
    data = request.json
    idx = data.get('index', 0)
    if 0 <= idx < len(chains):
        state.current_idx = idx
        state.load_chain()
        state.neck_buffer.clear()
        state.scale_buffer.clear()
        state.tilt_buffer.clear()
        return jsonify({'success': True, 'chain': chain_names[idx]})
    return jsonify({'success': False})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    if 'show_neck_dot' in data:
        state.show_neck_dot = data['show_neck_dot']
    if 'brightness_auto' in data:
        state.brightness_auto = data['brightness_auto']
    if 'chain_scale' in data:
        state.chain_scale = float(data['chain_scale'])
    if 'vertical_offset' in data:
        state.vertical_offset = float(data['vertical_offset'])
    return jsonify({'success': True})

def generate_frames():
    while True:
        ret, frame = state.cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        state.frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = state.face_mesh.process(rgb)
        h, w = frame.shape[:2]

        # Auto-brightness
        if state.brightness_auto:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            br = np.mean(gray)
            if br < 90:
                frame = cv2.convertScaleAbs(frame, alpha=1.7, beta=45)
            elif br > 180:
                frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=-20)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # Neck point
            neck = state.get_accurate_neck_position(lm, w, h)

            # Tilt & depth
            tilt, depth = state.get_head_tilt(lm, w, h)
            depth = state.prev_depth * 0.7 + depth * 0.3
            state.prev_depth = depth

            # Scale
            scale = state.calculate_optimal_scale(lm, w, h, state.chain_w, depth)

            # Resize & rotate
            new_w = int(state.chain_w * scale)
            new_h = int(state.chain_h * scale)
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(state.original_chain, (new_w, new_h))
                center = (new_w // 2, new_h // 2)
                M = cv2.getRotationMatrix2D(center, tilt, 1.0)

                cos = np.abs(M[0,0])
                sin = np.abs(M[0,1])
                new_w_rot = int((new_h * sin) + (new_w * cos))
                new_h_rot = int((new_h * cos) + (new_w * sin))

                M[0,2] += (new_w_rot / 2) - center[0]
                M[1,2] += (new_h_rot / 2) - center[1]

                rotated = cv2.warpAffine(resized, M, (new_w_rot, new_h_rot),
                                       borderValue=(0,0,0,0))

                face_height = np.linalg.norm(
                    np.array([lm[state.NOSE_TIP].x * w, lm[state.NOSE_TIP].y * h]) -
                    np.array([lm[state.CHIN].x * w, lm[state.CHIN].y * h])
                )
                hang_offset = int(face_height * 0.15)
                x = neck[0] - (new_w_rot // 2)
                y = neck[1] + hang_offset - int(new_h_rot * state.vertical_offset)

                frame = state.overlay_chain(frame, rotated, x, y)

            # Green dot
            if state.show_neck_dot and state.frame_count % 10 == 0:
                cv2.circle(frame, tuple(neck), 5, (0, 255, 0), -1)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Create templates folder and HTML file
    os.makedirs('templates', exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chain Fit Studio</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            color: #ffffff;
            overflow: hidden;
            height: 100vh;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        .control-panel {
            width: 350px;
            background: rgba(30, 30, 30, 0.95);
            backdrop-filter: blur(10px);
            padding: 30px;
            overflow-y: auto;
            box-shadow: 5px 0 20px rgba(0, 0, 0, 0.5);
        }

        .logo {
            text-align: center;
            margin-bottom: 30px;
        }

        .logo h1 {
            font-size: 28px;
            background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }

        .logo p {
            color: #888;
            font-size: 12px;
        }

        .section {
            background: rgba(40, 40, 40, 0.6);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #00ff88;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chain-list {
            max-height: 200px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 5px;
        }

        .chain-item {
            padding: 12px 15px;
            margin: 5px 0;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }

        .chain-item:hover {
            background: rgba(0, 255, 136, 0.1);
            border-color: rgba(0, 255, 136, 0.3);
        }

        .chain-item.active {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 204, 255, 0.2) 100%);
            border-color: #00ff88;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }

        .nav-btn {
            flex: 1;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
        }

        .nav-btn:hover {
            background: rgba(0, 255, 136, 0.2);
            border-color: #00ff88;
            transform: translateY(-2px);
        }

        .control-group {
            margin-bottom: 20px;
        }

        .control-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
            color: #ccc;
        }

        .control-value {
            color: #00ff88;
            font-weight: 600;
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0, 255, 136, 0.5);
            transition: all 0.3s ease;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: 0 4px 12px rgba(0, 255, 136, 0.8);
        }

        .toggle-group {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .toggle-group:last-child {
            border-bottom: none;
        }

        .toggle-label {
            font-size: 13px;
            color: #ccc;
        }

        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }

        .toggle-switch input {
            display: none;
        }

        .toggle-slider {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 26px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .toggle-slider:before {
            content: "";
            position: absolute;
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background: white;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        input:checked + .toggle-slider {
            background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
        }

        input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }

        .video-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #000;
            position: relative;
        }

        #videoFeed {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 0;
        }

        .status-bar {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-text {
            font-size: 14px;
            color: #00ff88;
        }

        .fps-counter {
            font-size: 12px;
            color: #888;
            font-family: 'Courier New', monospace;
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(0, 255, 136, 0.3);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 255, 136, 0.5);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .loading {
            animation: pulse 2s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="control-panel">
            <div class="logo">
                <h1>CHAIN FIT STUDIO</h1>
                <p>Real-time Virtual Try-On</p>
            </div>

            <div class="section">
                <div class="section-title">
                    <span>Chain Selection</span>
                </div>
                <div class="chain-list" id="chainList"></div>
                <div class="nav-buttons">
                    <button class="nav-btn" onclick="previousChain()">Previous</button>
                    <button class="nav-btn" onclick="nextChain()">Next</button>
                </div>
            </div>

            <div class="section">
                <div class="section-title">
                    <span>Adjustments</span>
                </div>
                
                <div class="control-group">
                    <div class="control-label">
                        <span>Chain Size</span>
                        <span class="control-value" id="scaleValue">1.0x</span>
                    </div>
                    <input type="range" id="chainScale" min="0.5" max="2.0" step="0.1" value="1.0">
                </div>

                <div class="control-group">
                    <div class="control-label">
                        <span>Vertical Position</span>
                        <span class="control-value" id="offsetValue">0.20</span>
                    </div>
                    <input type="range" id="verticalOffset" min="-0.3" max="0.5" step="0.05" value="0.20">
                </div>
            </div>

            <div class="section">
                <div class="section-title">
                    <span>Display Options</span>
                </div>
                
                <div class="toggle-group">
                    <span class="toggle-label">Show Neck Anchor</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="showNeckDot" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <div class="toggle-group">
                    <span class="toggle-label">Auto Brightness</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="autoBrightness" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>
        </div>

        <div class="video-panel">
            <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Feed">
            <div class="status-bar">
                <div class="status-text" id="statusText">
                    <span class="loading">LIVE</span>
                    <span id="currentChain"></span>
                </div>
                <div class="fps-counter">Streaming at 30 FPS</div>
            </div>
        </div>
    </div>

    <script>
        let chains = [];
        let currentIndex = 0;

        fetch('/api/chains')
            .then(r => r.json())
            .then(data => {
                chains = data.chains;
                currentIndex = data.current;
                renderChains();
                updateStatus();
            });

        function renderChains() {
            const list = document.getElementById('chainList');
            list.innerHTML = chains.map((chain, idx) => 
                `<div class="chain-item ${idx === currentIndex ? 'active' : ''}" 
                      onclick="selectChain(${idx})">
                    ${chain}
                </div>`
            ).join('');
        }

        function selectChain(index) {
            fetch('/api/chain/select', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index})
            }).then(() => {
                currentIndex = index;
                renderChains();
                updateStatus();
            });
        }

        function previousChain() {
            selectChain((currentIndex - 1 + chains.length) % chains.length);
        }

        function nextChain() {
            selectChain((currentIndex + 1) % chains.length);
        }

        function updateStatus() {
            document.getElementById('currentChain').textContent = 
                ' | ' + chains[currentIndex];
        }

        function updateSettings() {
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    show_neck_dot: document.getElementById('showNeckDot').checked,
                    brightness_auto: document.getElementById('autoBrightness').checked,
                    chain_scale: parseFloat(document.getElementById('chainScale').value),
                    vertical_offset: parseFloat(document.getElementById('verticalOffset').value)
                })
            });
        }

        document.getElementById('chainScale').addEventListener('input', (e) => {
            document.getElementById('scaleValue').textContent = e.target.value + 'x';
            updateSettings();
        });

        document.getElementById('verticalOffset').addEventListener('input', (e) => {
            document.getElementById('offsetValue').textContent = e.target.value;
            updateSettings();
        });

        document.getElementById('showNeckDot').addEventListener('change', updateSettings);
        document.getElementById('autoBrightness').addEventListener('change', updateSettings);

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') previousChain();
            if (e.key === 'ArrowRight') nextChain();
        });
    </script>
</body>
</html>"""
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n" + "="*50)
    print("CHAIN FIT STUDIO - WEB APP")
    print("="*50)
    print("\nServer starting...")
    print("Open your browser and go to: http://localhost:5000")
    print("The video stream is CONTINUOUS - no start/stop!")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)