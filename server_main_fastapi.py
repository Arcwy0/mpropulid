from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import uvicorn
import argparse
import os
import json

from server_queue_manager import check_pending_queue, add_request_to_pending
from server_process_images import process_and_save_images, initialize_model

app = FastAPI()
scheduler = AsyncIOScheduler()
model = None

def schedule_pending_check():
    try:
        pending_requests = check_pending_queue()
        if not pending_requests:
            print("No pending requests to process.")
            return

        for request in pending_requests:
            print(f"Processing request for {request['email']}")
            process_and_save_images(model)  # Pass the loaded model to the image processing function
    except Exception as e:
        print(f"Error in schedule_pending_check: {e}")

@app.on_event("startup")
async def startup_event():
    global model
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PuLID for FLUX.1-dev")
    parser.add_argument('--version', type=str, default='v0.9.1', help='version of the model', choices=['v0.9.0', 'v0.9.1'])
    parser.add_argument("--name", type=str, default="flux-dev", choices=['flux-dev'],
                        help="currently only support flux-dev")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use, for 24G GPUs")
    parser.add_argument("--fp8", action="store_true", help="use flux-dev-fp8 model")
    parser.add_argument("--onnx_provider", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="set onnx_provider to cpu (default gpu) can help reduce RAM usage, and when combined with"
                                "fp8 option, the peak RAM is under 15GB")
    parser.add_argument("--port", type=int, default=8080, help="Port to use")
    parser.add_argument("--dev", action='store_true', help="Development mode")
    parser.add_argument("--pretrained_model", type=str, help='for development')
    args = parser.parse_args()

    if args.aggressive_offload:
        args.offload = True

    try:
        # Initialize the model with parsed arguments
        model = initialize_model(args)
        print("Model successfully loaded.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    try:
        # Start the scheduler
        scheduler.start()
        scheduler.add_job(
            schedule_pending_check,
            trigger=IntervalTrigger(minutes=1)
        )
        print("Scheduler started successfully.")
    except Exception as e:
        print(f"Failed to start scheduler: {e}")

# Request model for the API endpoint
class PhotoTransformRequest(BaseModel):
    email: str
    firstName: str
    photoStyle: str
    role: str
    purpose: str
    ageRange: str
    photo: str  # Base64-encoded image

@app.post("/api/photo-transform")
async def photo_transform(request: PhotoTransformRequest):
    try:
        # Add timestamp and initial status to the request
        request_data = request.dict()
        request_data["timestamp"] = datetime.now().isoformat()
        request_data["status"] = "pending"
        
        # Add request to pending.json
        add_request_to_pending(request_data)
        print(f"Request added to queue for email: {request_data['email']}")
        
        return {"success": True, "message": "Request added to queue"}
    
    except Exception as e:
        print(f"Error in photo_transform endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "PuLID Server is running with model loaded"}

# Run the server
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"Error starting the server: {e}")