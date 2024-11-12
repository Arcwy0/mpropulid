import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Configuration
SMTP_SERVER = "smtp.yandex.ru"
SMTP_PORT = 587  # For TLS
EMAIL_ADDRESS = "magazinepro@yandex.ru"
APP_PASSWORD = "qnkosbmcrhwmzzuj"

def send_email_with_images(recipient_email, image_paths):
    subject = "Your Generated Images"
    body = "Please find the generated images attached."

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    for image_path in image_paths:
        with open(image_path, "rb") as img_file:
            img = MIMEImage(img_file.read())
            img.add_header("Content-Disposition", "attachment", filename=image_path.split("/")[-1])
            msg.attach(img)

    # Set up the SMTP server and send the email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Enable TLS
            server.login(EMAIL_ADDRESS, APP_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")