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
# 2. Configure the Server IP and Port in the addon's panel in Blender (View3D > Sidebar > Server Test).
#    Default is 127.0.0.1 and port 5000.
# 3. In Blender's 3D View, add an object (e.g., a Cube) and make sure it is selected (active).
# 4. Press "N" in the 3D View to open the side panel.
# 5. Find the "Server Test" tab/panel.
# 6. Enter the server's IP address and port if different from the defaults.
# 7. Click the "Send Object Name" button.
# 8. Open Blender's system console (Window > Toggle System Console) to see the server's response
#    or any error messages.

# Note on 'requests' library:
# This addon uses the 'requests' library to make HTTP requests.
# If 'requests' is not part of Blender's bundled Python environment,
# you might need to install it into Blender's Python.
# One common way is to find Blender's Python executable and run:
# path/to/blender/python.exe -m pip install requests
# Or bundle the library with your addon. For simplicity, we assume it's available or can be installed.

bl_info = {
    "name": "Scene Data Sender",
    "author": "AI Assistant",
    "version": (1, 0),
    "blender": (2, 80, 0),  # Minimum Blender version
    "location": "View3D > Sidebar > Server Test",
    "description": "Sends basic scene data (object names, types, transforms) to a server.",
    "warning": "",
    "doc_url": "",
    "category": "Development",
}

import bpy
from bpy.props import StringProperty, IntProperty
import json
import urllib.request # Using urllib instead of requests for built-in compatibility
import urllib.error

# No global SERVER_URL needed anymore, it will be constructed from scene properties.

class SCENE_OT_send_data(bpy.types.Operator):
    """Sends basic scene data to the server"""
    bl_idname = "scene.send_data"
    bl_label = "Send Scene Data"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        objects_data = []

        if not scene.objects:
            self.report({'WARNING'}, "No objects in the scene to send.")
            print("Blender Addon: No objects in the scene.")
            # We can still send an empty list if desired, or cancel.
            # For now, let's send an empty list.
            # return {'CANCELLED'}

        for obj in scene.objects:
            objects_data.append({
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation_euler": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            })

        payload = {"scene_objects": objects_data}
        
        # Convert payload to JSON string and then to bytes
        json_payload = json.dumps(payload).encode('utf-8')

        # Construct server URL from scene properties
        server_ip = scene.server_ip_address
        server_port = scene.server_port
        if not server_ip:
            self.report({'ERROR'}, "Server IP address is not set in the panel.")
            print("Blender Addon: Server IP address is not set.")
            return {'CANCELLED'}
        
        # Consider changing the endpoint on the server side as well.
        # For now, we'll keep /api/echo_name and the server will just acknowledge.
        # A better endpoint would be /api/scene_data
        server_url = f"http://{server_ip}:{server_port}/api/scene_data" # Changed endpoint

        self.report({'INFO'}, f"Sending scene data for {len(objects_data)} object(s) to {server_url}")
        print(f"Blender Addon: Sending data for {len(objects_data)} object(s) to {server_url}")
        if objects_data:
            print(f"Blender Addon: First object data: {objects_data[0]}")

        try:
            req = urllib.request.Request(server_url, data=json_payload, headers={'Content-Type': 'application/json'}, method='POST')
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

class SCENEDATA_PT_panel(bpy.types.Panel):
    """Creates a Panel in the 3D View Sidebar for sending scene data"""
    bl_label = "Scene Data Sender"
    bl_idname = "SCENEDATA_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Scene Sync' # Tab name in the N-panel

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Server Configuration:")
        row = box.row()
        row.prop(scene, "server_ip_address", text="IP Address")
        row = box.row()
        row.prop(scene, "server_port", text="Port")
        
        layout.separator()
        
        row = layout.row()
        row.operator(SCENE_OT_send_data.bl_idname)

# List of classes to register
classes = (
    SCENE_OT_send_data,
    SCENEDATA_PT_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.server_ip_address = StringProperty(
        name="Server IP Address",
        description="IP address of the Flask server",
        default="127.0.0.1"
    )
    bpy.types.Scene.server_port = IntProperty(
        name="Server Port",
        description="Port number of the Flask server",
        default=5000,
        min=1,
        max=65535
    )
    print("Blender Addon: 'Scene Data Sender' registered with properties.")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.server_ip_address
    del bpy.types.Scene.server_port
    print("Blender Addon: 'Scene Data Sender' unregistered with properties.")

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