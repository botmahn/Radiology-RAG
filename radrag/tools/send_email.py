import os
from typing import Dict, Any
from datetime import datetime
from langchain.tools import tool

#Email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

def send_email_with_pdf(
    recipient_email: str,
    pdf_path: str,
    patient_name: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    sender_email: str = None,
    sender_password: str = None
) -> str:
    """
    Send email with PDF report attached.
    
    Args:
        recipient_email: Recipient's email address
        pdf_path: Path to PDF file
        patient_name: Patient name for subject line
        smtp_server: SMTP server address
        smtp_port: SMTP port
        sender_email: Sender's email
        sender_password: Sender's password/app password
    
    Returns:
        Success/failure message
    """
    if not sender_email or not sender_password:
        return "Error: Email credentials not configured. Please set SENDER_EMAIL and SENDER_PASSWORD."
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Chest X-Ray Report - {patient_name}"
        
        # Email body
        body = f"""Dear Kendall Jenner's boyfriend,

Please find attached the chest X-ray radiology report for patient {patient_name}.

This report was generated using our automated medical imaging analysis system.

Best regards,
Radiology Department
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        with open(pdf_path, 'rb') as f:
            pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
            msg.attach(pdf_attachment)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return f"Email sent successfully to {recipient_email}"
    
    except Exception as e:
        return f"Error sending email: {str(e)}"