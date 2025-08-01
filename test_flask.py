from flask import Flask, Response, render_template, jsonify, request
import cv2

app = Flask(__name__)

# Switch states (initially all OFF)
switch_states = {"s0": 0, "s1": 0, "s2": 0}

# Open camera (change index if needed)
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Render the main page with switches and live video."""
    return render_template('index.html', switch_states=switch_states)

@app.route('/video_feed')
def video_feed():
    """Provide video stream."""
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_switch/<switch>', methods=['POST'])
def toggle_switch(switch):
    """Toggle the state of a switch."""
    if switch in switch_states:
        switch_states[switch] = 1 - switch_states[switch]  # Toggle between 0 and 1
        return jsonify(switch_states)
    return jsonify({"error": "Invalid switch"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
