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

# Main execution block to run the Flask development server
if __name__ == '__main__':
    # Run the app on all available IP addresses (0.0.0.0)
    # and on port 5000.
    # Debug mode can be enabled for development (app.run(host='0.0.0.0', port=5000, debug=True))
    # but should be turned off for production.
    app.run(host='0.0.0.0', port=5000)