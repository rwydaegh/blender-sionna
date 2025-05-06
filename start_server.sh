#!/bin/bash
fuser -k 5000/tcp || true
source venv/bin/activate
python3 server.py