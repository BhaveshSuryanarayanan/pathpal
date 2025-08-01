import asyncio
import json
import base64
import cv2
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed
import logging
import threading
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state
toggle_state = False
active_connections = set()
camera = None
camera_lock = threading.Lock()

def get_camera():
    """Initialize and return the camera object"""
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)  # 0 is usually the default camera
            if not camera.isOpened():
                logger.error("Failed to open camera")
                return None
            # Set resolution (optional, adjust as needed)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            return None
    return camera

def release_camera():
    """Safely release the camera resource"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            logger.info("Camera released")

def process_image(image, switch_state, processing_params=None):
    """
    Process an input image based on the switch state and optional parameters.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        switch_state: Boolean indicating the state of the toggle switch
        processing_params: Dictionary with additional processing parameters (optional)
            - brightness: float, brightness adjustment factor (default: 1.0)
            - contrast: float, contrast adjustment factor (default: 1.0)
            - blur_amount: int, amount of Gaussian blur to apply (default: 0)
            - edge_detection: bool, whether to apply edge detection (default: False)
    
    Returns:
        processed_image: Processed image as numpy array
        metadata: Dictionary with processing information
    """
    if processing_params is None:
        processing_params = {}
    
    # Make a copy to avoid modifying the original
    processed = image.copy()
    
    # Get parameters with defaults
    brightness = processing_params.get('brightness', 1.0)
    contrast = processing_params.get('contrast', 1.0)
    blur_amount = processing_params.get('blur_amount', 0)
    edge_detection = processing_params.get('edge_detection', False)
    
    metadata = {
        "applied_effects": [],
        "switch_state": switch_state
    }
    
    # Process based on switch state
    if switch_state:
        # When switch is ON - apply selected effects
        
        # Apply brightness and contrast adjustments
        processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness * 50)
        metadata["applied_effects"].append("brightness_contrast")
        
        # Apply blur if requested
        if blur_amount > 0:
            processed = cv2.GaussianBlur(processed, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
            metadata["applied_effects"].append("gaussian_blur")
        
        # Apply edge detection if requested
        if edge_detection:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.Canny(gray, 100, 200)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            metadata["applied_effects"].append("edge_detection")
    
    # Add timestamp and dimensions to metadata
    metadata["dimensions"] = f"{processed.shape[1]}x{processed.shape[0]}"
    
    return processed, metadata

def get_frame():
    """Capture a frame from the camera and return it as a base64 string"""
    with camera_lock:
        cam = get_camera()
        if cam is None:
            return None, None
        
        success, frame = cam.read()
        if not success:
            logger.error("Failed to read frame from camera")
            return None, None
        
        # Process the frame based on toggle state
        processing_params = {
            'brightness': 1.2 if toggle_state else 1.0,
            'contrast': 1.3 if toggle_state else 1.0,
            'blur_amount': 2 if toggle_state else 0,
            'edge_detection': toggle_state
        }
        
        processed_frame, metadata = process_image(frame, toggle_state, processing_params)
        
        # Convert to JPEG and encode as base64
        try:
            # Convert from BGR (OpenCV format) to RGB (PIL format)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Save to in-memory file
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=70)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_str, metadata
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None, None

async def handle_client(websocket):
    """Handle WebSocket client connection"""
    global toggle_state, active_connections
    
    # Register new client
    client_id = id(websocket)
    active_connections.add(websocket)
    logger.info(f"New client connected. Total clients: {len(active_connections)}")
    
    try:
        while True:
            # Wait for messages from the client
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get('type') == 'request_frame':
                # Client is requesting a new frame
                frame_data, metadata = get_frame()
                if frame_data:
                    await websocket.send(json.dumps({
                        'type': 'frame',
                        'frame': frame_data,
                        'metadata': metadata
                    }))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Failed to capture frame'
                    }))
            
            elif data.get('type') == 'toggle_state':
                # Client is updating the toggle state
                new_state = data.get('value', False)
                if new_state != toggle_state:
                    toggle_state = new_state
                    logger.info(f"Toggle state changed to: {toggle_state}")
                    
                    # Acknowledge the state change
                    await websocket.send(json.dumps({
                        'type': 'toggle_ack',
                        'value': toggle_state
                    }))
            
            elif data.get('type') == 'process_image':
                # Client is sending an image to process
                try:
                    # Get the image data and decode from base64
                    img_data = data.get('image', '')
                    if not img_data:
                        raise ValueError("No image data provided")
                    
                    # Get processing parameters
                    processing_params = data.get('processing_params', {})
                    
                    # Decode the base64 image
                    img_bytes = base64.b64decode(img_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        raise ValueError("Failed to decode image")
                    
                    # Process the image
                    processed_img, metadata = process_image(img, toggle_state, processing_params)
                    
                    # Encode processed image to base64
                    processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(processed_rgb)
                    buffer = BytesIO()
                    pil_img.save(buffer, format="JPEG", quality=70)
                    processed_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    # Send processed image back to client
                    await websocket.send(json.dumps({
                        'type': 'processed_image',
                        'image': processed_b64,
                        'metadata': metadata
                    }))
                    
                except Exception as e:
                    logger.error(f"Error processing uploaded image: {str(e)}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f'Error processing image: {str(e)}'
                    }))
    
    except ConnectionClosed:
        logger.info(f"Client disconnected")
    except Exception as e:
        logger.error(f"Error handling client: {str(e)}")
    finally:
        # Unregister the client
        active_connections.remove(websocket)
        logger.info(f"Client removed. Total clients: {len(active_connections)}")
        
        # Release camera if no more clients
        if len(active_connections) == 0:
            release_camera()

async def main():
    """Start the WebSocket server"""
    async with websockets.serve(
        handle_client,
        "localhost",
        8000,
        # Increase max message size for video frames
        max_size=10_485_760,  # 10MB
        # Increase timeouts
        ping_interval=30,
        ping_timeout=10
    ) as server:
        logger.info("WebSocket server started at ws://localhost:8000")
        # Keep the server running indefinitely
        await asyncio.Future()  # This will run forever until interrupted

if __name__ == "__main__":
    try:
        # Initialize camera on startup
        get_camera()
        
        # Start the server
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        # Ensure camera is released
        release_camera()