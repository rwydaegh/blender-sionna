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

# Main execution block to run the Flask development server
if __name__ == '__main__':
    # Run the app on all available IP addresses (0.0.0.0)
    # and on port 5000.
    # Debug mode can be enabled for development (app.run(host='0.0.0.0', port=5000, debug=True))
    # but should be turned off for production.
    app.run(host='0.0.0.0', port=5000)