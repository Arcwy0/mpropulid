import json
import os
from datetime import datetime

QUEUE_DIR = "/app/queue"
PENDING_FILE = os.path.join(QUEUE_DIR, "pending.json")
PROCESSED_FILE = os.path.join(QUEUE_DIR, "processed.json")

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{file_path} not found. Creating an empty file.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error reading {file_path}: {e}")
        return []

def write_json_file(file_path, data):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def check_pending_queue():
    os.makedirs(QUEUE_DIR, exist_ok=True)
    
    if not os.path.exists(PENDING_FILE):
        print("Creating pending.json file")
        write_json_file(PENDING_FILE, [])
        
    if not os.path.exists(PROCESSED_FILE):
        print("Creating processed.json file")
        write_json_file(PROCESSED_FILE, [])

    pending_queue = read_json_file(PENDING_FILE)
    pending_requests = [item for item in pending_queue if item.get('status') == 'pending']
    print(f"Found {len(pending_requests)} pending requests.")
    return pending_requests

def mark_request_processed(request):
    pending_queue = read_json_file(PENDING_FILE)
    for item in pending_queue:
        if item['email'] == request['email'] and item['timestamp'] == request['timestamp']:
            item['status'] = 'processed'
    
    write_json_file(PENDING_FILE, pending_queue)
    
    # Ensure processed.json exists
    processed_queue = read_json_file(PROCESSED_FILE)
    processed_queue.append(request)
    write_json_file(PROCESSED_FILE, processed_queue)
    print(f"Marked request as processed: {request['email']}")

def add_request_to_pending(request_data):
    os.makedirs(QUEUE_DIR, exist_ok=True)
    pending_queue = read_json_file(PENDING_FILE)
    pending_queue.append(request_data)
    write_json_file(PENDING_FILE, pending_queue)
    print(f"Added request to pending queue: {request_data['email']}")