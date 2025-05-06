# blender_addon.py
# Blender Addon Client

# Installation in Blender:
# 1. Save this file as blender_addon.py.
# 2. Open Blender.
# 3. Go to Edit > Preferences > Add-ons.
# 4. Click "Install..." and navigate to select blender_addon.py.
# 5. Enable the addon by checking the box next to its name ("Server Test Panel").

# How to Use:
# 1. Make sure the Flask server (server.py) is running.
# 2. If the server is on a remote machine (e.g., Vast.ai):
#    IMPORTANT: Edit the SERVER_URL in this script (around line 39)
#    to `http://<YOUR_SERVER_PUBLIC_IP>:<PORT>/api/echo_name`.
#    Replace <YOUR_SERVER_PUBLIC_IP> with the actual public IP of your server
#    and <PORT> with the externally accessible port for the Flask app.
# 3. In Blender's 3D View, add an object (e.g., a Cube) and make sure it is selected (active).
# 4. Press "N" in the 3D View to open the side panel.
# 5. Find the "Server Test" tab/panel.
# 6. Click the "Send Object Name" button.
# 7. Open Blender's system console (Window > Toggle System Console) to see the server's response
#    or any error messages.

# Note on 'requests' library:
# This addon uses the 'requests' library to make HTTP requests.
# If 'requests' is not part of Blender's bundled Python environment,
# you might need to install it into Blender's Python.
# One common way is to find Blender's Python executable and run:
# path/to/blender/python.exe -m pip install requests
# Or bundle the library with your addon. For simplicity, we assume it's available or can be installed.

bl_info = {
    "name": "Server Test Panel",
    "author": "AI Assistant",
    "version": (1, 0),
    "blender": (2, 80, 0),  # Minimum Blender version
    "location": "View3D > Sidebar > Server Test",
    "description": "Sends active object name to a server and prints response.",
    "warning": "",
    "doc_url": "",
    "category": "Development",
}

import bpy
import json
import urllib.request # Using urllib instead of requests for built-in compatibility
import urllib.error

# Configuration
# For local testing with server.py on the same machine:
SERVER_URL = "http://127.0.0.1:5000/api/echo_name"
# For remote server (e.g., Vast.ai):
# SERVER_URL = "http://YOUR_VAST_AI_PUBLIC_IP:PORT/api/echo_name" # <-- EDIT THIS FOR REMOTE SERVER

class OBJECT_OT_send_name(bpy.types.Operator):
    """Sends the active object's name to the server"""
    bl_idname = "object.send_active_object_name"
    bl_label = "Send Object Name"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        active_obj = context.active_object

        if active_obj is None:
            self.report({'ERROR'}, "No active object selected.")
            print("Blender Addon: No active object selected.")
            return {'CANCELLED'}

        object_name = active_obj.name
        payload = {"object_name": object_name}
        
        # Convert payload to JSON string and then to bytes
        json_payload = json.dumps(payload).encode('utf-8')

        self.report({'INFO'}, f"Sending object name: {object_name} to {SERVER_URL}")
        print(f"Blender Addon: Sending '{object_name}' to {SERVER_URL}")

        try:
            req = urllib.request.Request(SERVER_URL, data=json_payload, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=10) as response: # Added timeout
                response_data = response.read().decode('utf-8')
                response_json = json.loads(response_data)
                
                message = response_json.get("message", "No message field in response.")
                self.report({'INFO'}, f"Server response: {message}")
                print(f"Blender Addon: Server response: {message}")
                print(f"Blender Addon: Full server response: {response_data}")

        except urllib.error.URLError as e:
            error_message = f"Network error: {e.reason}"
            if hasattr(e, 'code'):
                error_message += f" (HTTP Status Code: {e.code})"
            self.report({'ERROR'}, error_message)
            print(f"Blender Addon: {error_message}")
            if hasattr(e, 'read'): # If there's a response body with the error
                try:
                    error_body = e.read().decode('utf-8')
                    print(f"Blender Addon: Server error details: {error_body}")
                except Exception as read_err:
                    print(f"Blender Addon: Could not read error body: {read_err}")
        except json.JSONDecodeError:
            self.report({'ERROR'}, "Failed to decode JSON response from server.")
            print("Blender Addon: Failed to decode JSON response from server.")
        except Exception as e:
            self.report({'ERROR'}, f"An unexpected error occurred: {str(e)}")
            print(f"Blender Addon: An unexpected error occurred: {str(e)}")
            
        return {'FINISHED'}

class SERVERTEST_PT_panel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Server Test"
    bl_idname = "SERVERTEST_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Server Test' # Tab name in the N-panel

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator(OBJECT_OT_send_name.bl_idname)

# List of classes to register
classes = (
    OBJECT_OT_send_name,
    SERVERTEST_PT_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    print("Blender Addon: 'Server Test Panel' registered.")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Blender Addon: 'Server Test Panel' unregistered.")

if __name__ == "__main__":
    # This part is for testing registration from Blender's text editor
    # To actually use the addon, install it through Preferences > Add-ons
    # and enable it.
    #
    # If you run this script directly in Blender's text editor:
    # 1. Make sure to unregister first if it was previously registered from the editor.
    # try:
    #     unregister()
    # except Exception as e:
    #     pass # Ignore if not registered
    # register()
    pass