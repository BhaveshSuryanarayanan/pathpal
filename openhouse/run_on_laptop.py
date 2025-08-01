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
import time
import argparse

from mqtt_module import MQTT
from midas_module import Midas
from grocery_picking_module import Grocery_picking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state variables
toggle_state = False
active_connections = set()
current_frame = None
frame_lock = threading.Lock()
running = True

# Parse command line arguments
parser = argparse.ArgumentParser(description="Control output display")
parser.add_argument("--display", "-d", action="store_true", help="Display output")
parser.add_argument("--port", "-p", type=int, default=8000, help="WebSocket server port")
args = parser.parse_args()

def get_frame_for_websocket():
    """Get the current frame for WebSocket transmission"""
    global current_frame
    
    with frame_lock:
        if current_frame is None:
            return None, None
        frame = current_frame.copy()
    
    # Convert to JPEG and encode as base64
    try:
        # Convert from BGR (OpenCV format) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Save to in-memory file
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=70)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        metadata = {
            "timestamp": time.time(),
            "toggle_state": toggle_state
        }
        
        return img_str, metadata
    except Exception as e:
        logger.error(f"Error processing frame for websocket: {str(e)}")
        return None, None

async def handle_client(websocket):
    """Handle WebSocket client connection"""
    global toggle_state, active_connections
    
    # Register new client
    active_connections.add(websocket)
    logger.info(f"New client connected. Total clients: {len(active_connections)}")
    
    try:
        while True:
            # Wait for messages from the client
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get('type') == 'request_frame':
                # Client is requesting a new frame
                frame_data, metadata = get_frame_for_websocket()
                if frame_data:
                    await websocket.send(json.dumps({
                        'type': 'frame',
                        'frame': frame_data,
                        'metadata': metadata
                    }))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'No frame available'
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
    
    except ConnectionClosed:
        logger.info(f"Client disconnected")
    except Exception as e:
        logger.error(f"Error handling client: {str(e)}")
    finally:
        # Unregister the client
        active_connections.remove(websocket)
        logger.info(f"Client removed. Total clients: {len(active_connections)}")

async def run_websocket_server():
    """Run the WebSocket server"""
    server = await websockets.serve(
        handle_client,
        "localhost",
        args.port,
        max_size=10_485_760,  # 10MB
        ping_interval=30,
        ping_timeout=10
    )
    
    logger.info(f"WebSocket server started at ws://localhost:{args.port}")
    
    try:
        # Keep the server running until the program exits
        while running:
            await asyncio.sleep(1)
    finally:
        server.close()
        await server.wait_closed()
        logger.info("WebSocket server stopped")

def inference_loop():
    """Run the main inference loop in a separate thread"""
    global current_frame, running
    
    logger.info("Starting inference loop")
    
    # Initialize models and MQTT
    broker = "test.mosquitto.org"
    port = 1883
    send_topic = "vibrator/matrix"
    receive_topic = "vibrator/switch_state"

    try:
        midas_model = Midas('m')
        gp_model = Grocery_picking() 

        mqtt_obj = MQTT(broker=broker, port=port, send_topic=send_topic, receive_topic=receive_topic)
        mqtt_obj.initialize_connection()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            running = False
            return
            
        logger.info("Camera opened successfully")
        
        while running:
            ret, frame = cap.read()
            
            if not ret:
                logger.error('Failed to grab frame')
                time.sleep(0.1)
                continue
            
            # Copy the original frame for websocket
            with frame_lock:
                current_frame = frame.copy()
            
            switch = toggle_state
            print(toggle_state)
            try:
                # s0, s1, s2 = mqtt_obj.get_switch_state()
                # logger.debug(f"Switch states: {s0}, {s1}, {s2}")
                
                # if s0 == 0 and s2 == 0:
                #     switch = 1
                # else:
                #     switch = 2
                    
                if switch:
                    midas_model.predict(frame)
                    midas_model.locate_obstacles()
                    mat = midas_model.find_matrix()
                    
                    if args.display:
                        # Update the display frame
                        display_frame = midas_model.display(show=False)
                        with frame_lock:
                            current_frame = display_frame
                            current_frame = np.ones(display_frame.shape)
                    # mqtt_obj.send_matrix(mat, show=False)
                    
                else:
                    mat = gp_model.predict(frame)
                    
                    if args.display:
                        # Update the display frame
                        display_frame = gp_model.display(frame, show=False)
                        with frame_lock:
                            current_frame = display_frame

                    # mqtt_obj.send_matrix(mat, show=True)
            except Exception as e:
                logger.error(f"Error in inference processing: {str(e)}")
                
            # Check for keyboard interrupt
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
                
            # Small delay to prevent CPU hogging
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Inference loop error: {str(e)}")
    finally:
        logger.info("Closing camera")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Inference loop stopped")

async def main():
    """Main function to run both websocket server and inference loop"""
    global running
    
    # Start inference in a separate thread
    inference_thread = threading.Thread(target=inference_loop)
    inference_thread.daemon = True
    inference_thread.start()
    
    # Run websocket server in the main asyncio loop
    try:
        await run_websocket_server()
    finally:
        # Ensure clean shutdown
        running = False
        logger.info("Waiting for inference thread to finish...")
        inference_thread.join(timeout=5.0)
        logger.info("Shutdown complete")

if __name__ == "__main__":
    try:
        # Run the main asyncio loop
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        running = False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Ensure everything is cleaned up
        running = False
        cv2.destroyAllWindows()