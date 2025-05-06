# server.py
# Flask Backend Server

# Installation:
# Open your terminal or command prompt.
# Install Flask using pip:
# pip install Flask

# Running the server:
# Navigate to the directory where this file (server.py) is saved.
# Run the script using:
# python server.py
# The server will start, typically on http://0.0.0.0:5000/

from flask import Flask, request, jsonify
import sionna.rt # Import sionna.rt
from sionna.rt import Transmitter, Receiver, PlanarArray # Import necessary Sionna classes

# Create a Flask application instance
app = Flask(__name__)

# Define the API endpoint /api/echo_name that accepts POST requests
@app.route('/api/echo_name', methods=['POST'])
def echo_name():
    """
    Receives a JSON payload with "object_name",
    and returns a JSON response echoing the name.
    """
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if 'object_name' is in the received data
        if data and 'object_name' in data:
            object_name = data['object_name']
            # Prepare the response message
            response_message = f"Server received: {object_name}"
            print(f"Received object_name: {object_name}") # Log to server console
            return jsonify({"message": response_message}), 200
        else:
            # If 'object_name' is not found, return an error response
            print("Error: 'object_name' not found in request.")
            return jsonify({"error": "Missing 'object_name' in JSON payload"}), 400
    except Exception as e:
        # Handle any other errors during request processing
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

# Define the API endpoint /api/scene_data that accepts POST requests
@app.route('/api/scene_data', methods=['POST'])
def scene_data():
    """
    Receives a JSON payload with "scene_objects" (a list of object data),
    and returns a JSON response acknowledging the data.
    """
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if 'scene_objects' is in the received data and is a list
        if data and 'scene_objects' in data and isinstance(data['scene_objects'], list):
            scene_objects = data['scene_objects']
            num_objects = len(scene_objects)
            
            # Log to server console
            print(f"Received data for {num_objects} scene object(s).")
            if num_objects > 0:
                # Log details of the first object as an example
                print(f"First object details: {scene_objects[0]}")
            
            # Prepare the response message
            response_message = f"Server received data for {num_objects} object(s)."
            return jsonify({"message": response_message, "objects_received": num_objects}), 200
        else:
            # If 'scene_objects' is not found or not a list, return an error response
            error_msg = "Missing 'scene_objects' list in JSON payload."
            if data and 'scene_objects' in data:
                error_msg = "'scene_objects' must be a list."
            print(f"Error: {error_msg}")
            return jsonify({"error": error_msg}), 400
    except Exception as e:
        # Handle any other errors during request processing
        print(f"Error processing /api/scene_data request: {e}")
        return jsonify({"error": str(e)}), 500
    
    # Define the API endpoint /setup_sionna_scene that accepts POST requests
    @app.route('/setup_sionna_scene', methods=['POST'])
    def setup_sionna_scene():
        """
        Receives XML file path and Tx/Rx data, sets up Sionna scene,
        and performs minimal validation.
        """
        try:
            # Get the JSON data from the request
            data = request.get_json()
    
            # Extract xml_path and tx_rx_list
            xml_path = data.get('xml_path')
            tx_rx_list = data.get('tx_rx_list', [])
    
            if not xml_path:
                print("Error: 'xml_path' not found in request.")
                return jsonify({"status": "error", "message": "Missing 'xml_path' in JSON payload"}), 400
    
            print(f"Received request to setup Sionna scene with XML: {xml_path}")
            print(f"Received {len(tx_rx_list)} Tx/Rx objects data.")
    
            # --- Sionna Scene Setup Logic ---
            try:
                # Load the Mitsuba scene
                scene = sionna.rt.load_scene(xml_path)
                print(f"Sionna scene loaded from {xml_path}")
    
                # Set default antenna arrays (can be refined later)
                scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
                scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
                print("Default antenna arrays set.")
    
                # Add Transmitters and Receivers from received data
                added_tx = []
                added_rx = []
                for obj_data in tx_rx_list:
                    obj_type = obj_data.get("blender_name", "").split("_")[0] # Simple type inference from name prefix
                    sionna_name = obj_data.get("custom_properties", {}).get("sionna_name", obj_data.get("blender_name"))
                    location = obj_data.get("location")
    
                    if not sionna_name or not location:
                        print(f"Skipping Tx/Rx object due to missing name or location: {obj_data}")
                        continue
    
                    position = location # Sionna expects position as a list/array
    
                    if obj_type == "TX":
                        tx = Transmitter(name=sionna_name, position=position)
                        scene.add(tx)
                        added_tx.append(sionna_name)
                        print(f"Added Sionna Transmitter: {sionna_name} at {position}")
                    elif obj_type == "RX":
                        rx = Receiver(name=sionna_name, position=position)
                        scene.add(rx)
                        added_rx.append(sionna_name)
                        print(f"Added Sionna Receiver: {sionna_name} at {position}")
                    else:
                        print(f"Skipping object with unrecognized prefix: {obj_data.get('blender_name')}")
    
    
                # Minimal validation
                validation_output = {
                    "scene_objects_count": len(scene.objects),
                    "transmitters_added": added_tx,
                    "receivers_added": added_rx,
                    "total_transmitters_in_scene": len(scene.transmitters),
                    "total_receivers_in_scene": len(scene.receivers)
                }
                print("Sionna scene setup complete. Validation output:")
                print(validation_output)
    
                return jsonify({"status": "success", "message": "Sionna scene setup complete.", "details": validation_output}), 200
    
            except Exception as sionna_e:
                print(f"Error during Sionna scene setup: {sionna_e}")
                return jsonify({"status": "error", "message": f"Error during Sionna scene setup: {str(sionna_e)}", "details": str(sionna_e)}), 500
    
        except Exception as e:
            print(f"Error processing /setup_sionna_scene request: {e}")
            return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}", "details": str(e)}), 500
    
    
    # Main execution block to run the Flask development server
    if __name__ == '__main__':
        # Run the app on all available IP addresses (0.0.0.0)
        # and on port 5000.
        # Debug mode can be enabled for development (app.run(host='0.0.0.0', port=5000, debug=True))
        # but should be turned off for production.
        app.run(host='0.0.0.0', port=5000)