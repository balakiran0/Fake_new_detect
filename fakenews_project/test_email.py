#!/usr/bin/env python
"""
Email Test Script for Fake News Detector
Run this script to test email functionality and diagnose issues.
"""

import os
import sys
import django
from pathlib import Path

# Add the project directory to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fakenews_project.settings')
django.setup()

from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.models import User
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def test_django_email():
    """Test Django's email functionality"""
    print("🧪 Testing Django Email Configuration...")
    print(f"📧 Email Backend: {settings.EMAIL_BACKEND}")
    print(f"📧 SMTP Host: {settings.EMAIL_HOST}")
    print(f"📧 SMTP Port: {settings.EMAIL_PORT}")
    print(f"📧 Use TLS: {settings.EMAIL_USE_TLS}")
    print(f"📧 From Email: {settings.DEFAULT_FROM_EMAIL}")
    print("-" * 50)
    
    try:
        # Test email
        test_email = input("Enter test email address: ").strip()
        if not test_email:
            print("❌ No email provided. Exiting.")
            return False
            
        subject = 'Test Email from Fake News Detector'
        message = '''
        🎉 Congratulations! 
        
        Your email configuration is working correctly.
        
        This is a test email sent from your Fake News Detector application.
        
        If you received this email, your SMTP settings are properly configured.
        
        Best regards,
        Fake News Detector Team
        '''
        
        send_mail(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [test_email],
            fail_silently=False,
        )
        
        print(f"✅ Test email sent successfully to {test_email}")
        print("📬 Please check your inbox (and spam folder)")
        return True
        
    except Exception as e:
        print(f"❌ Email sending failed: {str(e)}")
        return False

def test_smtp_connection():
    """Test direct SMTP connection"""
    print("\n🔌 Testing Direct SMTP Connection...")
    
    try:
        server = smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT)
        server.starttls()
        server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
        print("✅ SMTP connection successful")
        server.quit()
        return True
    except Exception as e:
        print(f"❌ SMTP connection failed: {str(e)}")
        return False

def test_password_reset_email():
    """Test password reset email specifically"""
    print("\n🔐 Testing Password Reset Email...")
    
    try:
        from django.contrib.auth.forms import PasswordResetForm
        from django.contrib.sites.models import Site
        
        # Get or create a test user
        email = input("Enter email address to test password reset: ").strip()
        if not email:
            print("❌ No email provided. Exiting.")
            return False
            
        user, created = User.objects.get_or_create(
            email=email,
            defaults={'username': email.split('@')[0]}
        )
        
        if created:
            print(f"✅ Created test user: {email}")
        else:
            print(f"✅ Using existing user: {email}")
        
        # Create password reset form and send email
        form = PasswordResetForm({'email': email})
        if form.is_valid():
            form.save(
                request=None,
                use_https=False,
                from_email=settings.DEFAULT_FROM_EMAIL,
                email_template_name='registration/password_reset_email.html',
                subject_template_name='registration/password_reset_subject.txt',
            )
            print(f"✅ Password reset email sent to {email}")
            print("📬 Please check your inbox (and spam folder)")
            return True
        else:
            print(f"❌ Form validation failed: {form.errors}")
            return False
            
    except Exception as e:
        print(f"❌ Password reset email failed: {str(e)}")
        return False

def diagnose_common_issues():
    """Diagnose common email issues"""
    print("\n🔍 Diagnosing Common Email Issues...")
    
    issues_found = []
    
    # Check Gmail app password
    if '@gmail.com' in settings.EMAIL_HOST_USER:
        print("📧 Gmail detected. Checking configuration...")
        if len(settings.EMAIL_HOST_PASSWORD.replace(' ', '')) != 16:
            issues_found.append("Gmail App Password should be 16 characters (without spaces)")
        else:
            print("✅ Gmail App Password format looks correct")
    
    # Check TLS settings
    if settings.EMAIL_PORT == 587 and not settings.EMAIL_USE_TLS:
        issues_found.append("Port 587 requires EMAIL_USE_TLS = True")
    elif settings.EMAIL_PORT == 465:
        issues_found.append("Port 465 requires EMAIL_USE_SSL = True (not EMAIL_USE_TLS)")
    
    # Check from email format
    if not settings.DEFAULT_FROM_EMAIL:
        issues_found.append("DEFAULT_FROM_EMAIL is not set")
    
    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print("✅ No obvious configuration issues found")
    
    return len(issues_found) == 0

def main():
    """Main test function"""
    print("=" * 60)
    print("📧 FAKE NEWS DETECTOR - EMAIL DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Run diagnostics
    config_ok = diagnose_common_issues()
    smtp_ok = test_smtp_connection()
    
    if smtp_ok:
        django_ok = test_django_email()
        if django_ok:
            reset_ok = test_password_reset_email()
        else:
            reset_ok = False
    else:
        django_ok = False
        reset_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Configuration Check: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"SMTP Connection:     {'✅ PASS' if smtp_ok else '❌ FAIL'}")
    print(f"Django Email:        {'✅ PASS' if django_ok else '❌ FAIL'}")
    print(f"Password Reset:      {'✅ PASS' if reset_ok else '❌ FAIL'}")
    
    if not smtp_ok:
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Check if Gmail 2FA is enabled and you're using an App Password")
        print("2. Verify the App Password is correct (16 characters)")
        print("3. Check if 'Less secure app access' is enabled (if not using App Password)")
        print("4. Try using console backend for testing: EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'")
        print("5. Check firewall/antivirus blocking SMTP connections")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
