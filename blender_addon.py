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
import os # Import the os module
import logging # Import the logging module
import traceback # Import the traceback module

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

class SCENE_OT_prepare_sionna_export(bpy.types.Operator):
    """Prepares scene data and exports for Sionna simulation"""
    bl_idname = "scene.prepare_sionna_export"
    bl_label = "Prepare Sionna Export"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.report({'INFO'}, "Starting Sionna export preparation...")
        logging.info("Blender Addon: Starting Sionna export preparation...")

        scene = context.scene
        temp_dir = bpy.path.abspath("//temp/") # Use a relative path within the blend file directory
        os.makedirs(temp_dir, exist_ok=True)
        xml_filepath = os.path.join(temp_dir, "sionna_scene.xml")

        # 1. Call Mitsuba export operator
        try:
            # Ensure the Mitsuba exporter is available and enabled
            if not hasattr(bpy.ops.export_scene, "mitsuba"):
                 self.report({'ERROR'}, "Mitsuba exporter not found. Please ensure the Mitsuba Blender addon is installed and enabled.")
                 logging.error("Blender Addon: Mitsuba exporter not found.")
                 return {'CANCELLED'}

            logging.info(f"Blender Addon: Attempting to call Mitsuba export operator with filepath='{xml_filepath}', export_ids=True")
            bpy.ops.export_scene.mitsuba(filepath=xml_filepath, export_ids=True) # Reverted to implicit context
            logging.info("Blender Addon: Mitsuba export operator call finished.")
            self.report({'INFO'}, f"Mitsuba scene exported to {xml_filepath}")
            logging.info(f"Blender Addon: Mitsuba scene exported to {xml_filepath}")

        except Exception as e:
            tb_str = traceback.format_exc()
            self.report({'ERROR'}, f"Failed to export Mitsuba scene: {str(e)}. Traceback: {tb_str}")
            logging.error(f"Blender Addon: Failed to export Mitsuba scene: {str(e)}")
            logging.error(f"Blender Addon Traceback: {tb_str}")
            return {'CANCELLED'}

        # 2. Find Tx/Rx Empties and collect data
        tx_rx_data = []
        for obj in scene.objects:
            # Check if the object is an Empty and its name starts with TX_ or RX_
            if obj.type == 'EMPTY' and (obj.name.startswith("TX_") or obj.name.startswith("RX_")):
                obj_data = {
                    "blender_name": obj.name,
                    "location": [obj.location.x, obj.location.y, obj.location.z],
                    "rotation_euler": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                    "custom_properties": {}
                }
                # Collect custom properties starting with 'sionna_'
                for key in obj.keys():
                    if key.startswith("sionna_"):
                        obj_data["custom_properties"][key] = obj[key]
                tx_rx_data.append(obj_data)
                logging.info(f"Blender Addon: Collected data for {obj.name}: {obj_data}")

        if not tx_rx_data:
            self.report({'WARNING'}, "No objects starting with 'TX_' or 'RX_' found in the scene.")
            logging.warning("Blender Addon: No objects starting with 'TX_' or 'RX_' found.")
            # Decide if we should still send the XML without Tx/Rx data or cancel
            # For now, let's proceed and send an empty tx_rx_data list

        # 3. Package and send data to server (This will be implemented in the next step)
        # For now, just report the collected data
        self.report({'INFO'}, f"Collected data for {len(tx_rx_data)} Tx/Rx objects.")
        logging.info(f"Blender Addon: Collected data for {len(tx_rx_data)} Tx/Rx objects: {tx_rx_data}")


        # 3. Read XML file content and package data for server
        try:
            with open(xml_filepath, 'r', encoding='utf-8') as f:
                xml_file_content = f.read()
            logging.info(f"Blender Addon: Successfully read XML content from {xml_filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read XML file {xml_filepath}: {str(e)}")
            logging.error(f"Blender Addon: Failed to read XML file {xml_filepath}: {str(e)}", exc_info=True)
            return {'CANCELLED'}

        payload = {
            "xml_content": xml_file_content, # Send content instead of path
            "tx_rx_list": tx_rx_data
        }

        json_payload = json.dumps(payload).encode('utf-8')

        # Construct server URL from scene properties
        server_ip = scene.server_ip_address
        server_port = scene.server_port
        if not server_ip:
            self.report({'ERROR'}, "Server IP address is not set in the panel.")
            print("Blender Addon: Server IP address is not set.")
            return {'CANCELLED'}

        # Use the new endpoint for setting up the Sionna scene
        server_url = f"http://{server_ip}:{server_port}/setup_sionna_scene"

        self.report({'INFO'}, f"Sending scene setup data to {server_url}")
        logging.info(f"Blender Addon: Sending scene setup data to {server_url}")
        # logging.info(f"Blender Addon: Payload: {payload}") # Uncomment for debugging

        try:
            req = urllib.request.Request(server_url, data=json_payload, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response: # Increased timeout for potential server processing
                response_data = response.read().decode('utf-8')
                response_json = json.loads(response_data)

                message = response_json.get("message", "No message field in response.")
                status = response_json.get("status", "unknown")
                details = response_json.get("details", "No details provided.")

                self.report({'INFO'}, f"Server response: Status: {status}, Message: {message}")
                logging.info(f"Blender Addon: Server response: Status: {status}, Message: {message}")
                logging.info(f"Blender Addon: Server response details: {details}")


        except urllib.error.URLError as e:
            detailed_error_message = f"Network error: {e.reason}" # Renamed from error_message for clarity
            if hasattr(e, 'code'):
                detailed_error_message += f" (HTTP Status Code: {e.code})"
            
            server_details_str = "" # Store details from server response
            if hasattr(e, 'read'): # If there's a response body with the error
                try:
                    # Attempt to read and decode the error body from the server response
                    error_body_content = e.read().decode('utf-8')
                    # The error_body_content is often JSON from our Flask server, containing the actual error
                    server_details_str = f" Server Details: {error_body_content}"
                    # Log the raw server details; exc_info=False because error_body_content is the primary info here
                    logging.error(f"Blender Addon: Server error details: {error_body_content}", exc_info=False)
                except Exception as read_err:
                    server_details_str = " Server Details: Could not read or decode error body from server."
                    logging.error(f"Blender Addon: Could not read server error body: {read_err}", exc_info=True)
            
            # Combine the network error with server details for reporting
            final_report_message = detailed_error_message + server_details_str
            
            # Blender's self.report has a limited display length, so truncate if necessary for the UI.
            # The full message will be in the logs.
            max_blender_report_length = 250 # Adjust if necessary, Blender UI might truncate very long messages
            if len(final_report_message) > max_blender_report_length:
                report_to_ui = final_report_message[:max_blender_report_length - 3] + "..."
            else:
                report_to_ui = final_report_message

            self.report({'ERROR'}, report_to_ui)
            # Log the full combined error message along with client-side traceback for comprehensive debugging
            logging.error(f"Blender Addon: Full error context: {final_report_message}", exc_info=True)
        except json.JSONDecodeError:
            self.report({'ERROR'}, "Failed to decode JSON response from server.")
            logging.error("Blender Addon: Failed to decode JSON response from server.", exc_info=True) # Added exc_info=True
        except Exception as e:
            self.report({'ERROR'}, f"An unexpected error occurred: {str(e)}")
            logging.error(f"Blender Addon: An unexpected error occurred: {str(e)}", exc_info=True) # Added exc_info=True


        self.report({'INFO'}, "Sionna export preparation and data sent.")
        logging.info("Blender Addon: Sionna export preparation and data sent.")

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

        # Existing button for sending basic data
        row = layout.row()
        row.operator(SCENE_OT_send_data.bl_idname)

        layout.separator() # Add a separator for clarity

        # New button for Phase 2 Sionna export
        row = layout.row()
        row.operator(SCENE_OT_prepare_sionna_export.bl_idname)


# List of classes to register
classes = (
    SCENE_OT_send_data,
    SCENE_OT_prepare_sionna_export, # Add the new operator here
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