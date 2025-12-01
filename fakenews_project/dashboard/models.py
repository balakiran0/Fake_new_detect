import uuid
import random
import string
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
import json


class EmailConfirmationToken(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.UUIDField(default=uuid.uuid4, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    confirmed_at = models.DateTimeField(null=True, blank=True)
    
    def is_expired(self):
        """Check if token is expired (24 hours)"""
        return timezone.now() > self.created_at + timedelta(hours=24)
    
    def confirm(self):
        """Mark token as confirmed and activate user"""
        self.confirmed_at = timezone.now()
        self.user.is_active = True
        self.user.save()
        self.save()
    
    def __str__(self):
        return f"Email confirmation for {self.user.email}"


class EmailOTP(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    email = models.EmailField()
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    verified_at = models.DateTimeField(null=True, blank=True)
    attempts = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-created_at']
    
    @classmethod
    def generate_otp(cls):
        """Generate a 6-digit OTP"""
        return ''.join(random.choices(string.digits, k=6))
    
    def is_expired(self):
        """Check if OTP is expired (10 minutes)"""
        return timezone.now() > self.created_at + timedelta(minutes=10)
    
    def is_valid(self):
        """Check if OTP is valid (not expired, not verified, attempts < 3)"""
        return not self.is_expired() and not self.verified_at and self.attempts < 3
    
    def verify(self, entered_otp):
        """Verify the OTP and activate user if correct"""
        self.attempts += 1
        self.save()
        
        if self.attempts >= 3:
            return False, "Too many failed attempts. Please request a new OTP."
        
        if self.is_expired():
            return False, "OTP has expired. Please request a new one."
        
        if self.otp == entered_otp:
            self.verified_at = timezone.now()
            self.user.is_active = True
            self.user.save()
            self.save()
            return True, "Email verified successfully!"
        else:
            return False, f"Invalid OTP. {3 - self.attempts} attempts remaining."
    
    def __str__(self):
        return f"OTP for {self.email} - {self.otp}"


class Conversation(models.Model):
    """Model to store chat conversations like ChatGPT"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_archived = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.title or 'Untitled Conversation'} - {self.user.username}"
    
    def get_title(self):
        """Generate title from first message if not set"""
        if self.title:
            return self.title
        
        first_message = self.messages.filter(role='user').first()
        if first_message:
            # Take first 50 characters of the message
            title = first_message.content[:50]
            if len(first_message.content) > 50:
                title += "..."
            return title
        return "New Conversation"
    
    def get_message_count(self):
        """Get total number of messages in conversation"""
        return self.messages.count()
    
    def get_last_message_time(self):
        """Get timestamp of last message"""
        last_message = self.messages.last()
        return last_message.created_at if last_message else self.created_at


class Message(models.Model):
    """Model to store individual messages in conversations"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    MESSAGE_TYPE_CHOICES = [
        ('text', 'Text'),
        ('analysis', 'Analysis'),
        ('file_upload', 'File Upload'),
        ('error', 'Error'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPE_CHOICES, default='text')
    metadata = models.JSONField(default=dict, blank=True)  # For storing analysis results, file info, etc.
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
    
    def is_analysis_result(self):
        """Check if message contains analysis results"""
        return self.message_type == 'analysis' and 'analysis_result' in self.metadata
    
    def get_analysis_data(self):
        """Get analysis data from metadata"""
        if self.is_analysis_result():
            return self.metadata.get('analysis_result', {})
        return {}


class UserPreferences(models.Model):
    """Model to store user preferences for chat interface"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='chat_preferences')
    theme = models.CharField(max_length=20, default='dark', choices=[
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('system', 'System'),
    ])
    accent_color = models.CharField(max_length=20, default='pink', choices=[
        ('pink', 'Pink'),
        ('blue', 'Blue'),
        ('green', 'Green'),
        ('purple', 'Purple'),
    ])
    language = models.CharField(max_length=20, default='en-US', choices=[
        ('en-US', 'English (US)'),
        ('en-GB', 'English (UK)'),
        ('es-ES', 'Spanish'),
    ])
    auto_save_conversations = models.BooleanField(default=True)
    show_timestamps = models.BooleanField(default=True)
    enable_sound_effects = models.BooleanField(default=True)
    max_conversations = models.IntegerField(default=50)  # Limit conversations per user
    
    def __str__(self):
        return f"Preferences for {self.user.username}"
