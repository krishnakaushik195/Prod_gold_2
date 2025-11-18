"""
Chain Fit Studio - Web App with Flask (Browser Camera Version)
This version processes video in the BROWSER, not on the server
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import base64
import mediapipe as mp
import math

app = Flask(__name__)

# ================== CONFIG ==================
CHAIN_FOLDER = "chains"
os.makedirs(CHAIN_FOLDER, exist_ok=True)

# Load chains and convert to base64
chains_data = []
chain_names = []

print("Loading chains from 'chains/'...")
for file in sorted(os.listdir(CHAIN_FOLDER)):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(CHAIN_FOLDER, file)
        with open(path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
            chains_data.append({
                'name': os.path.splitext(file)[0],
                'data': f"data:image/png;base64,{img_data}"
            })
            chain_names.append(os.path.splitext(file)[0])
            print(f"   Loaded: {file}")

if not chains_data:
    raise FileNotFoundError("Error: Put chain PNGs in 'chains/' folder!")

# ================== ROUTES ==================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chains')
def get_chains():
    return jsonify({
        'chains': chains_data,
        'names': chain_names
    })

@app.after_request
def add_security_headers(response):
    response.headers['Permissions-Policy'] = 'camera=*'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

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

        .video-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #000;
            position: relative;
        }

        #videoElement {
            display: none;
        }

        #canvasElement {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
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

        .start-button {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            padding: 20px 40px;
            font-size: 18px;
            background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
            border: none;
            border-radius: 12px;
            color: #000;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 24px rgba(0, 255, 136, 0.4);
        }

        .start-button:hover {
            transform: translate(-50%, -50%) scale(1.05);
            box-shadow: 0 12px 32px rgba(0, 255, 136, 0.6);
        }

        .error-message {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 50, 50, 0.9);
            padding: 20px 40px;
            border-radius: 12px;
            text-align: center;
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
                    <span>🔗 Chain Selection</span>
                </div>
                <div class="chain-list" id="chainList"></div>
                <div class="nav-buttons">
                    <button class="nav-btn" onclick="previousChain()">← Previous</button>
                    <button class="nav-btn" onclick="nextChain()">Next →</button>
                </div>
            </div>

            <div class="section">
                <div class="section-title">
                    <span>⚙️ Adjustments</span>
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
        </div>

        <div class="video-panel">
            <button class="start-button" id="startButton">🎥 Start Camera</button>
            <video id="videoElement" autoplay playsinline></video>
            <canvas id="canvasElement"></canvas>
            <div class="status-bar">
                <div class="status-text">
                    <span id="statusText">Click Start Camera</span>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>

    <script>
        let chains = [];
        let currentIndex = 0;
        let chainScale = 1.0;
        let verticalOffset = 0.20;
        let faceMesh;
        let camera;
        let currentChainImage = new Image();
        let isProcessing = false;

        const video = document.getElementById('videoElement');
        const canvas = document.getElementById('canvasElement');
        const ctx = canvas.getContext('2d');

        // Load chains from server
        fetch('/api/chains')
            .then(r => r.json())
            .then(data => {
                chains = data.chains;
                renderChainList();
                loadChain(0);
            });

        function renderChainList() {
            const list = document.getElementById('chainList');
            list.innerHTML = chains.map((chain, idx) => 
                `<div class="chain-item ${idx === currentIndex ? 'active' : ''}" 
                      onclick="selectChain(${idx})">
                    ${chain.name}
                </div>`
            ).join('');
        }

        function selectChain(index) {
            currentIndex = index;
            loadChain(index);
            renderChainList();
            updateStatus();
        }

        function loadChain(index) {
            currentChainImage.src = chains[index].data;
        }

        function previousChain() {
            selectChain((currentIndex - 1 + chains.length) % chains.length);
        }

        function nextChain() {
            selectChain((currentIndex + 1) % chains.length);
        }

        function updateStatus() {
            document.getElementById('statusText').textContent = 
                '🟢 LIVE | ' + chains[currentIndex].name;
        }

        // Start camera
        document.getElementById('startButton').addEventListener('click', async () => {
            try {
                document.getElementById('startButton').style.display = 'none';
                
                // Initialize MediaPipe Face Mesh
                faceMesh = new FaceMesh({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
                });
                
                faceMesh.setOptions({
                    maxNumFaces: 1,
                    refineLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });
                
                faceMesh.onResults(onResults);

                // Start camera
                camera = new Camera(video, {
                    onFrame: async () => {
                        if (!isProcessing) {
                            isProcessing = true;
                            await faceMesh.send({image: video});
                            isProcessing = false;
                        }
                    },
                    width: 1280,
                    height: 720
                });
                
                await camera.start();
                updateStatus();
                
            } catch (err) {
                alert('Camera access denied! Please allow camera permission and refresh the page.');
                console.error('Camera error:', err);
            }
        });

        function onResults(results) {
            canvas.width = results.image.width;
            canvas.height = results.image.height;

            // Draw video frame
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

            if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                const landmarks = results.multiFaceLandmarks[0];
                drawChain(landmarks, canvas.width, canvas.height);
            }
        }

        function drawChain(landmarks, w, h) {
            // Key landmarks
            const JAW_LEFT = 234;
            const JAW_RIGHT = 454;
            const CHIN = 152;
            const NOSE_TIP = 4;

            const jawL = {x: landmarks[JAW_LEFT].x * w, y: landmarks[JAW_LEFT].y * h};
            const jawR = {x: landmarks[JAW_RIGHT].x * w, y: landmarks[JAW_RIGHT].y * h};
            const chin = {x: landmarks[CHIN].x * w, y: landmarks[CHIN].y * h};
            const nose = {x: landmarks[NOSE_TIP].x * w, y: landmarks[NOSE_TIP].y * h};

            // Calculate neck position
            const jawMidX = (jawL.x + jawR.x) / 2;
            const jawMidY = (jawL.y + jawR.y) / 2;
            const faceLength = Math.sqrt((nose.x - chin.x)**2 + (nose.y - chin.y)**2);
            const jawWidth = Math.sqrt((jawR.x - jawL.x)**2 + (jawR.y - jawL.y)**2);

            let verticalOff = faceLength * 0.3;
            const widthFactor = Math.min(jawWidth / w, 0.3);
            verticalOff += widthFactor * faceLength * 0.8;

            const neckX = jawMidX;
            const neckY = chin.y + verticalOff;

            // Calculate scale
            const targetW = jawWidth * (1.4 + (jawWidth / w) * 0.8);
            let scale = targetW / currentChainImage.width;
            scale *= chainScale;

            // Calculate rotation
            const angle = Math.atan2(jawR.y - jawL.y, jawR.x - jawL.x);

            // Draw chain
            const chainW = currentChainImage.width * scale;
            const chainH = currentChainImage.height * scale;

            ctx.save();
            ctx.translate(neckX, neckY + (faceLength * 0.15) - (chainH * verticalOffset));
            ctx.rotate(angle);
            ctx.drawImage(currentChainImage, -chainW/2, 0, chainW, chainH);
            ctx.restore();
        }

        // Controls
        document.getElementById('chainScale').addEventListener('input', (e) => {
            chainScale = parseFloat(e.target.value);
            document.getElementById('scaleValue').textContent = chainScale.toFixed(1) + 'x';
        });

        document.getElementById('verticalOffset').addEventListener('input', (e) => {
            verticalOffset = parseFloat(e.target.value);
            document.getElementById('offsetValue').textContent = verticalOffset.toFixed(2);
        });

        // Keyboard shortcuts
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
    print("CHAIN FIT STUDIO - BROWSER CAMERA VERSION")
    print("="*50)
    print("\nServer starting...")
    print("Open your browser and go to: http://localhost:5000")
    print("Click 'Start Camera' button to begin!")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
