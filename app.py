"""
Chain Fit Studio - Flask Backend (Browser Camera Version)
This version ONLY serves the API - HTML is already in templates/
"""

from flask import Flask, render_template, jsonify
import os
import base64

app = Flask(__name__)

# ================== CONFIG ==================
CHAIN_FOLDER = "chains"
os.makedirs(CHAIN_FOLDER, exist_ok=True)

# Load chains and convert to base64
chains_data = []
chain_names = []

print("="*50)
print("CHAIN FIT STUDIO - Loading chains...")
print("="*50)

for file in sorted(os.listdir(CHAIN_FOLDER)):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(CHAIN_FOLDER, file)
        try:
            with open(path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
                # Detect image type
                ext = file.lower().split('.')[-1]
                mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg'] else "image/png"
                
                chains_data.append({
                    'name': os.path.splitext(file)[0],
                    'data': f"data:{mime_type};base64,{img_data}"
                })
                chain_names.append(os.path.splitext(file)[0])
                print(f"   ✓ Loaded: {file}")
        except Exception as e:
            print(f"   ✗ Error loading {file}: {e}")

if not chains_data:
    print("\n⚠️  WARNING: No chain images found in 'chains/' folder!")
    print("   Please add PNG/JPG images to the 'chains/' folder.\n")
else:
    print(f"\n✓ Total chains loaded: {len(chains_data)}\n")

print("="*50)

# ================== ROUTES ==================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/chains')
def get_chains():
    """API endpoint to get all chain images as base64"""
    return jsonify({
        'chains': chains_data,
        'names': chain_names
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'chains_loaded': len(chains_data)
    })

# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers for camera access"""
    response.headers['Permissions-Policy'] = 'camera=*'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# ================== MAIN ==================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("CHAIN FIT STUDIO - SERVER STARTING")
    print("="*50)
    print("\n🌐 Server will be available at:")
    print("   Local: http://localhost:5000")
    print("   Production: https://your-app.onrender.com")
    print("\n📝 Features:")
    print("   ✓ Browser-based camera processing")
    print("   ✓ Real-time face mesh detection")
    print("   ✓ Virtual chain try-on")
    print("\n" + "="*50 + "\n")
    
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        debug=False,
        threaded=True,
        host='0.0.0.0',
        port=port
    )
