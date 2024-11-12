from flask import Flask, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import os
import json
from celery import Celery
from process_and_email_old import process_request  # Updated import

# Initialize Flask and enable CORS
app = Flask(__name__)
CORS(app)

# Directory paths
QUEUE_DIR = "queue"
PENDING_FILE = os.path.join(QUEUE_DIR, "pending.json")
PROCESSED_FILE = os.path.join(QUEUE_DIR, "processed.json")

# Initialize Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # Redis as the message broker
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Ensure necessary files and folders
def ensure_queue_files():
    os.makedirs(QUEUE_DIR, exist_ok=True)
    for file_path in [PENDING_FILE, PROCESSED_FILE]:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Queue processing function
@celery.task
def process_pending_requests():
    ensure_queue_files()
    pending_requests = read_json_file(PENDING_FILE)
    
    if not pending_requests:
        print("No pending requests to process.")
        return
    
    processed_requests = read_json_file(PROCESSED_FILE)

    for request_data in pending_requests:
        # Asynchronously process each request
        try:
            process_request.delay(request_data)  # Use Celery's delay to queue each task
            request_data['status'] = 'processed'
            request_data['processed_timestamp'] = datetime.now().isoformat()
            processed_requests.append(request_data)
            print(f"Queued request for processing: {request_data['email']}")
        except Exception as e:
            print(f"Failed to queue request: {e}")
            request_data['status'] = 'failed'

    # Clear pending requests and save updated processed list
    write_json_file(PENDING_FILE, [])
    write_json_file(PROCESSED_FILE, processed_requests)

# Schedule the periodic task every 5 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(func=process_pending_requests, trigger="interval", minutes=5)
scheduler.start()

# API Endpoint to test the server
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"status": "Server is running and queue processing every 5 minutes."})

if __name__ == '__main__':
    ensure_queue_files()
    app.run(port=5000)