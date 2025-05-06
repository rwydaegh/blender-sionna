#!/bin/bash
fuser -k 5000/tcp || true
cd blender-sionna
source venv/bin/activate
python3 server.py