from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from app_flux import FluxGenerator, SamplingOptions
from datetime import datetime
import json
import os
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Initialize FastAPI app
app = FastAPI()

# Directories and file paths within Docker
QUEUE_DIR = "/app/queue"  # Ensure consistent paths within Docker
PENDING_FILE = os.path.join(QUEUE_DIR, "pending.json")
PROCESSED_FILE = os.path.join(QUEUE_DIR, "processed.json")
PROCESSED_IMAGES_DIR = "/app/processed_images"

# Create directories if they do not exist
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(QUEUE_DIR, exist_ok=True)

# Model loading (initialize once at startup to save GPU memory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "flux-dev"
flux_generator = FluxGenerator(model_name=model_name, device=device, offload=False, aggressive_offload=False, args={})

# Define request schema with Pydantic
class PhotoTransformRequest(BaseModel):
    email: str
    photoStyle: str
    photo: str  # base64-encoded image data
    timestamp: Optional[str] = None

# Utility functions for file management
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Function to process each request using the model
async def process_request(request_data):
    try:
        # Decode the image and prepare it for processing
        photo_style = request_data['photoStyle']
        email = request_data['email']
        photo_data = base64.b64decode(request_data['photo'])
        timestamp = request_data['timestamp']
        photo_path = os.path.join(PROCESSED_IMAGES_DIR, f"{timestamp}.jpg")
        
        with open(photo_path, "wb") as img_file:
            img_file.write(photo_data)

        # Configure sampling options
        sampling_options = SamplingOptions(
            prompt=photo_style,
            width=896,
            height=1152,
            num_steps=20,
            guidance=4,
            seed=-1
        )

        # Generate the image
        result_img, seed_used, _ = flux_generator.generate_image(
            width=sampling_options.width,
            height=sampling_options.height,
            num_steps=sampling_options.num_steps,
            start_step=0,
            guidance=sampling_options.guidance,
            seed=sampling_options.seed,
            prompt=sampling_options.prompt,
            id_image=photo_path
        )

        # Save processed image
        processed_path = os.path.join(PROCESSED_IMAGES_DIR, f"processed_{timestamp}.jpg")
        result_img.save(processed_path)

        # Update processed file and send email with image
        request_data['status'] = 'processed'
        request_data['processed_image_path'] = processed_path
        update_processed_file(request_data)
        await send_email_with_image(email, processed_path)

    except Exception as e:
        print(f"Error processing request: {e}")

# Endpoint to add new photo transformation request
@app.post("/api/photo-transform")
async def photo_transform(request: PhotoTransformRequest, background_tasks: BackgroundTasks):
    request_data = request.dict()
    request_data['timestamp'] = datetime.now().isoformat()
    request_data['status'] = 'pending'

    # Add to pending queue and start background processing
    update_pending_file(request_data)
    background_tasks.add_task(process_request, request_data)

    return {"success": True, "message": "Request added to queue for processing"}

# Endpoint to process all queued requests (useful if processing on demand)
@app.get("/api/process")
async def process_queue(background_tasks: BackgroundTasks):
    pending_queue = read_json_file(PENDING_FILE)

    for request_data in pending_queue:
        if request_data['status'] != 'processed':
            background_tasks.add_task(process_request, request_data)

    return {"success": True, "message": "All pending requests are being processed."}

# Update pending.json with new request
def update_pending_file(request_data):
    pending_queue = read_json_file(PENDING_FILE)
    pending_queue.append(request_data)
    write_json_file(PENDING_FILE, pending_queue)

# Update processed.json once request is complete
def update_processed_file(request_data):
    processed_queue = read_json_file(PROCESSED_FILE)
    processed_queue.append(request_data)
    write_json_file(PROCESSED_FILE, processed_queue)

# Email function
async def send_email_with_image(email, image_path):
    smtp_server = "smtp.example.com"  # Replace with actual SMTP server
    sender_email = "your-email@example.com"  # Replace with your email
    password = "your_password"  # Replace with your password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = "Your Processed Image"

    msg.attach(MIMEText("Here is your processed image.", 'plain'))
    with open(image_path, 'rb') as img_file:
        img = MIMEImage(img_file.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(img)

    server = smtplib.SMTP(smtp_server, 587)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, email, msg.as_string())
    server.quit()