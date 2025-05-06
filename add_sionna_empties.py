import bpy

# Function to add an Empty object with custom properties
def add_sionna_empty(name, location, custom_props):
    # Create an empty object
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=location)
    empty_obj = bpy.context.object
    empty_obj.name = name

    # Add custom properties
    for prop_name, prop_value in custom_props.items():
        empty_obj[prop_name] = prop_value

    print(f"Added Empty: {name} at {location} with properties {custom_props}")
    return empty_obj

# --- Add TX Empty ---
tx_name = "TX_Default"
tx_location = (0, 0, 5) # Example location
tx_props = {
    "sionna_name": "tx_from_blender",
    "sionna_pattern": "iso",
    "sionna_polarization": "V",
    "sionna_power_dbm": 20.0
}
add_sionna_empty(tx_name, tx_location, tx_props)

# --- Add RX Empty ---
rx_name = "RX_Default"
rx_location = (5, 5, 1) # Example location
rx_props = {
    "sionna_name": "rx_from_blender",
    "sionna_pattern": "iso",
    "sionna_polarization": "V"
}
add_sionna_empty(rx_name, rx_location, rx_props)

print("Sionna Empties script finished.")