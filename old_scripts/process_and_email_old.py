import os
import smtplib
import torch
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from PIL import Image
from flux.sampling import SamplingOptions
from pulid.pipeline_flux import PuLIDPipeline
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables for email configuration
load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Directory for saving processed images
PROCESSED_IMAGES_DIR = "processed_images"
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

# Initialize AI model (example setup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "your_model_name_here"  # Replace with your actual model identifier
pulid_model = PuLIDPipeline(model_name, device=device)
pulid_model.to(device)

def process_and_save_image(data):
    """Processes an image based on 'photoStyle' and saves it."""
    # Load image and style
    image_path = data['photo']
    photo_style = data['photoStyle']
    
    # Open the image file
    input_image = Image.open(image_path)
    
    # Prepare model options based on photo style and other custom settings
    options = SamplingOptions(
        prompt=photo_style,  # prompt to drive the style of the image
        width=input_image.width,
        height=input_image.height,
        num_steps=50,  # Number of inference steps
        guidance=7.5   # Guidance scale for image fidelity
    )
    
    # Generate the processed image using PuLID's pipeline
    processed_image, _, _ = pulid_model.generate_image(
        width=options.width,
        height=options.height,
        num_steps=options.num_steps,
        start_step=0,           # Start step (use 0 for full range)
        guidance=options.guidance,
        seed=None,              # Random seed for variability
        prompt=options.prompt
    )
    
    # Save processed image with a unique filename
    output_filename = f"{data['email']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    output_path = os.path.join(PROCESSED_IMAGES_DIR, output_filename)
    processed_image.save(output_path)
    
    return output_path

def send_email_with_attachment(email, subject, body, attachment_path):
    """Sends an email with the processed image attached."""
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = email
    msg['Subject'] = subject
    
    # Attach the body with MIMEText
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach the image file
    with open(attachment_path, "rb") as attachment:
        mime_base = MIMEBase('application', 'octet-stream')
        mime_base.set_payload(attachment.read())
        encoders.encode_base64(mime_base)
        mime_base.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
        msg.attach(mime_base)

    # Send email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, email, msg.as_string())

def process_request(request_data):
    """Processes a single request: runs model inference and sends email."""
    email = request_data['email']
    subject = "Your Processed Photo"
    body = f"Hello {request_data['firstName']},\n\nHere is your photo processed with the style '{request_data['photoStyle']}' as requested."

    try:
        # Process image and save to file
        processed_image_path = process_and_save_image(request_data)
        
        # Send email with the processed image
        send_email_with_attachment(email, subject, body, processed_image_path)
        print(f"Email sent to {email} with processed image.")
    
    except Exception as e:
        print(f"Error processing request for {email}: {e}")