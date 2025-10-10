from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordResetDoneView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, CreateView
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout
from django.urls import reverse_lazy
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.views import View
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .forms import CustomSignUpForm, CustomPasswordResetForm
from .models import EmailConfirmationToken, EmailOTP, Conversation, Message, UserPreferences

class DashboardLoginView(LoginView):
    template_name = "dashboard/login.html"
    redirect_authenticated_user = False  # Allow authenticated users to see login page
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['show_logout'] = self.request.user.is_authenticated
        return context


class DashboardHomeView(LoginRequiredMixin, TemplateView):
    template_name = "index.html"  # Use the main dashboard interface
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Try to get user preferences, but don't fail if tables don't exist yet
        try:
            preferences, created = UserPreferences.objects.get_or_create(
                user=self.request.user,
                defaults={
                    'theme': 'dark',
                    'auto_save_conversations': True,
                    'show_timestamps': True,
                    'enable_sound_effects': True,
                    'max_conversations': 50
                }
            )
            
            # Get user's conversations
            conversations = Conversation.objects.filter(
                user=self.request.user,
                is_archived=False
            ).order_by('-updated_at')[:20]  # Show last 20 conversations
            
            context.update({
                'conversations': conversations,
                'preferences': preferences,
                'has_conversations': conversations.exists(),
            })
        except Exception as e:
            # If tables don't exist yet, provide defaults
            context.update({
                'conversations': [],
                'preferences': None,
                'has_conversations': False,
            })
        
        return context


class CustomLogoutView(View):
    """Custom logout view that ensures proper redirect"""
    
    def get(self, request):
        return self.logout_user(request)
    
    def post(self, request):
        return self.logout_user(request)
    
    def logout_user(self, request):
        if request.user.is_authenticated:
            logout(request)
            messages.success(request, 'You have been logged out successfully.')
        return redirect('dashboard:login')


class SignUpView(CreateView):
    form_class = CustomSignUpForm
    template_name = "dashboard/signup.html"
    
    def post(self, request, *args, **kwargs):
        print(f"DEBUG: POST data received: {request.POST}")
        return super().post(request, *args, **kwargs)
    
    def form_valid(self, form):
        try:
            # Create user but keep inactive
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            print(f"DEBUG: User created - {user.email}")
            
            # Generate and send OTP
            otp_code = EmailOTP.generate_otp()
            print(f"DEBUG: Generated OTP - {otp_code}")
            
            otp = EmailOTP.objects.create(
                user=user,
                email=user.email,
                otp=otp_code
            )
            print(f"DEBUG: OTP saved to database - {otp.id}")
            
            # Send OTP email
            self.send_otp_email(user, otp_code)
            
            # Store user ID in session for OTP verification
            self.request.session['pending_user_id'] = user.id
            print(f"DEBUG: Session set for user ID - {user.id}")
            
            messages.success(
                self.request, 
                f'Account created! We\'ve sent a 6-digit OTP to {user.email}. Please verify to activate your account.'
            )
            return redirect('dashboard:verify_otp')
        except Exception as e:
            print(f"DEBUG: Error in form_valid - {str(e)}")
            messages.error(self.request, f'Error creating account: {str(e)}')
            return self.form_invalid(form)
    
    def send_otp_email(self, user, otp_code):
        """Send OTP email to user"""
        subject = 'Your Fake News Detector Verification Code'
        print(f"DEBUG: Preparing to send REAL email to {user.email} with OTP {otp_code}")
        
        # Create email content
        html_message = render_to_string('dashboard/emails/otp_email.html', {
            'user': user,
            'otp': otp_code,
        })
        plain_message = strip_tags(html_message)
        print(f"DEBUG: Email content prepared")
        
        try:
            # Send real email to user's actual email address
            send_mail(
                subject,
                plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [user.email],  # This sends to the user's actual email
                html_message=html_message,
                fail_silently=False,
            )
            print(f"DEBUG: REAL email sent successfully to {user.email}")
            messages.success(self.request, f'OTP sent to your email: {user.email}')
        except Exception as e:
            print(f"DEBUG: Email sending failed - {str(e)}")
            # Fallback: show OTP in message for testing
            messages.warning(self.request, f'Email service unavailable. Your OTP is: {otp_code}')
            messages.error(self.request, f'Failed to send email: {str(e)}')
    
    def form_invalid(self, form):
        print(f"DEBUG: Form is invalid - {form.errors}")
        for field, errors in form.errors.items():
            for error in errors:
                messages.error(self.request, f'{field}: {error}')
        return super().form_invalid(form)
    
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('dashboard:home')
        return super().dispatch(request, *args, **kwargs)


class VerifyOTPView(View):
    template_name = "dashboard/verify_otp.html"
    
    def get(self, request):
        # Check if there's a pending user
        user_id = request.session.get('pending_user_id')
        if not user_id:
            messages.error(request, 'No pending verification found. Please sign up again.')
            return redirect('dashboard:signup')
        
        try:
            user = User.objects.get(id=user_id)
            return render(request, self.template_name, {'user': user})
        except User.DoesNotExist:
            messages.error(request, 'Invalid verification session. Please sign up again.')
            return redirect('dashboard:signup')
    
    def post(self, request):
        user_id = request.session.get('pending_user_id')
        if not user_id:
            messages.error(request, 'No pending verification found. Please sign up again.')
            return redirect('dashboard:signup')
        
        try:
            user = User.objects.get(id=user_id)
            entered_otp = request.POST.get('otp', '').strip()
            
            if not entered_otp:
                messages.error(request, 'Please enter the OTP.')
                return render(request, self.template_name, {'user': user})
            
            # Get the latest valid OTP for this user
            otp_obj = EmailOTP.objects.filter(
                user=user,
                verified_at__isnull=True
            ).first()
            
            if not otp_obj:
                messages.error(request, 'No valid OTP found. Please request a new one.')
                return render(request, self.template_name, {'user': user})
            
            # Verify OTP
            success, message = otp_obj.verify(entered_otp)
            
            if success:
                # Clear session
                del request.session['pending_user_id']
                messages.success(request, message)
                return redirect('dashboard:login')
            else:
                messages.error(request, message)
                return render(request, self.template_name, {'user': user})
                
        except User.DoesNotExist:
            messages.error(request, 'Invalid verification session. Please sign up again.')
            return redirect('dashboard:signup')


class ResendOTPView(View):
    def post(self, request):
        user_id = request.session.get('pending_user_id')
        if not user_id:
            messages.error(request, 'No pending verification found. Please sign up again.')
            return redirect('dashboard:signup')
        
        try:
            user = User.objects.get(id=user_id)
            
            # Generate new OTP
            otp_code = EmailOTP.generate_otp()
            otp = EmailOTP.objects.create(
                user=user,
                email=user.email,
                otp=otp_code
            )
            
            # Send OTP email
            self.send_otp_email(user, otp_code)
            
            messages.success(request, f'New OTP sent to {user.email}')
            return redirect('dashboard:verify_otp')
            
        except User.DoesNotExist:
            messages.error(request, 'Invalid verification session. Please sign up again.')
            return redirect('dashboard:signup')
    
    def send_otp_email(self, user, otp_code):
        """Send OTP email to user"""
        subject = 'Your Fake News Detector Verification Code'
        
        # Create email content
        html_message = render_to_string('dashboard/emails/otp_email.html', {
            'user': user,
            'otp': otp_code,
        })
        plain_message = strip_tags(html_message)
        
        try:
            send_mail(
                subject,
                plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                html_message=html_message,
                fail_silently=False,
            )
        except Exception as e:
            messages.error(self.request, 'Failed to send OTP email. Please try again.')


def confirm_email(request, token):
    """Handle email confirmation (legacy)"""
    try:
        confirmation_token = get_object_or_404(EmailConfirmationToken, token=token)
        
        if confirmation_token.is_expired():
            messages.error(request, 'Confirmation link has expired. Please sign up again.')
            return redirect('dashboard:signup')
        
        if confirmation_token.confirmed_at:
            messages.info(request, 'Email already confirmed. You can now log in.')
            return redirect('dashboard:login')
        
        # Confirm the email
        confirmation_token.confirm()
        messages.success(request, 'Email confirmed successfully! You can now log in.')
        return redirect('dashboard:login')
        
    except Exception as e:
        messages.error(request, 'Invalid confirmation link.')
        return redirect('dashboard:signup')


class CustomPasswordResetView(PasswordResetView):
    """
    Custom password reset view with enhanced validation and error handling
    """
    form_class = CustomPasswordResetForm
    template_name = 'registration/password_reset_form.html'
    success_url = reverse_lazy('dashboard:password_reset_done')
    email_template_name = 'registration/password_reset_email.html'
    subject_template_name = 'registration/password_reset_subject.txt'
    
    def form_valid(self, form):
        """
        Send the password reset email using the current request host instead of
        the Sites framework, and then redirect to success_url.
        """
        email = form.cleaned_data['email']

        # Clear any existing messages to prevent accumulation
        storage = messages.get_messages(self.request)
        storage.used = True

        # Build options to ensure correct domain/protocol are used
        domain = getattr(settings, 'PASSWORD_RESET_DOMAIN', None) or self.request.get_host() or '127.0.0.1:8000'
        use_https = self.request.is_secure()

        form.save(
            domain_override=domain,
            use_https=use_https,
            from_email=settings.DEFAULT_FROM_EMAIL,
            email_template_name='registration/password_reset_email.txt',
            html_email_template_name=self.email_template_name,
            subject_template_name=self.subject_template_name,
            request=self.request,
            extra_email_context={
                'protocol': 'https' if use_https else 'http',
                'domain': domain,
                'base_url': ('https' if use_https else 'http') + '://' + domain,
                'site_name': 'Fake News Detector',
            },
        )

        # Log the request
        print(f"Password reset requested for: {email} (domain: {domain}, https: {use_https})")
        # No success message - keeping only the beautiful UI

        return HttpResponseRedirect(self.get_success_url())
    
    def get_context_data(self, **kwargs):
        """
        Override to ensure correct domain is used in email
        """
        context = super().get_context_data(**kwargs)
        # Ensure we use the correct protocol and domain
        context['protocol'] = 'http'
        context['domain'] = self.request.get_host()
        return context
    
    def form_invalid(self, form):
        """
        Override to add custom error handling
        """
        # Log the failed attempt
        if 'email' in form.cleaned_data:
            email = form.cleaned_data['email']
            print(f"Password reset failed for: {email}")
        
        return super().form_invalid(form)


class CustomPasswordResetDoneView(PasswordResetDoneView):
    """
    Custom password reset done view that clears old messages
    """
    template_name = 'registration/password_reset_done.html'
    
    def get(self, request, *args, **kwargs):
        # Clear all old messages completely
        storage = messages.get_messages(request)
        storage.used = True
        
        return super().get(request, *args, **kwargs)


class ProfileView(LoginRequiredMixin, TemplateView):
    """
    Profile view showing user account details and logout option
    """
    template_name = "dashboard/profile.html"
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Get user details
        context.update({
            'user': user,
            'full_name': user.get_full_name() or user.username,
            'email': user.email,
            'username': user.username,
            'date_joined': user.date_joined,
            'last_login': user.last_login,
            'is_staff': user.is_staff,
            'is_superuser': user.is_superuser,
        })
        
        return context


# Conversation Management Views

@method_decorator(csrf_exempt, name='dispatch')
class ConversationListView(LoginRequiredMixin, View):
    """API view to get user's conversations"""
    
    def get(self, request):
        conversations = Conversation.objects.filter(
            user=request.user,
            is_archived=False
        ).order_by('-updated_at')
        
        conversations_data = []
        for conv in conversations:
            conversations_data.append({
                'id': str(conv.id),
                'title': conv.get_title(),
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat(),
                'message_count': conv.get_message_count(),
                'last_message_time': conv.get_last_message_time().isoformat(),
            })
        
        return JsonResponse({
            'success': True,
            'conversations': conversations_data
        })


@method_decorator(csrf_exempt, name='dispatch')
class ConversationDetailView(LoginRequiredMixin, View):
    """API view to get conversation messages"""
    
    def get(self, request, conversation_id):
        try:
            conversation = Conversation.objects.get(
                id=conversation_id,
                user=request.user
            )
            
            messages = conversation.messages.all()
            messages_data = []
            
            for msg in messages:
                message_data = {
                    'id': str(msg.id),
                    'role': msg.role,
                    'content': msg.content,
                    'message_type': msg.message_type,
                    'created_at': msg.created_at.isoformat(),
                }
                
                # Add analysis data if available
                if msg.is_analysis_result():
                    message_data['analysis_data'] = msg.get_analysis_data()
                
                messages_data.append(message_data)
            
            return JsonResponse({
                'success': True,
                'conversation': {
                    'id': str(conversation.id),
                    'title': conversation.get_title(),
                    'created_at': conversation.created_at.isoformat(),
                    'updated_at': conversation.updated_at.isoformat(),
                },
                'messages': messages_data
            })
            
        except Conversation.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Conversation not found'
            }, status=404)


@method_decorator(csrf_exempt, name='dispatch')
class CreateConversationView(LoginRequiredMixin, View):
    """API view to create new conversation"""
    
    def post(self, request):
        try:
            # Create new conversation
            conversation = Conversation.objects.create(
                user=request.user,
                title=""  # Will be auto-generated from first message
            )
            
            return JsonResponse({
                'success': True,
                'conversation_id': str(conversation.id),
                'message': 'New conversation created'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class DeleteConversationView(LoginRequiredMixin, View):
    """API view to delete conversation"""
    
    def post(self, request, conversation_id):
        try:
            conversation = Conversation.objects.get(
                id=conversation_id,
                user=request.user
            )
            conversation.delete()
            
            return JsonResponse({
                'success': True,
                'message': 'Conversation deleted successfully'
            })
            
        except Conversation.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Conversation not found'
            }, status=404)


@method_decorator(csrf_exempt, name='dispatch')
class SaveMessageView(LoginRequiredMixin, View):
    """API view to save messages to conversation"""
    
    def post(self, request):
        try:
            import json
            data = json.loads(request.body)
            
            conversation_id = data.get('conversation_id')
            role = data.get('role')  # 'user' or 'assistant'
            content = data.get('content')
            message_type = data.get('message_type', 'text')
            metadata = data.get('metadata', {})
            
            if not all([conversation_id, role, content]):
                return JsonResponse({
                    'success': False,
                    'error': 'Missing required fields'
                }, status=400)
            
            # Get or create conversation
            if conversation_id == 'new':
                conversation = Conversation.objects.create(
                    user=request.user,
                    title=""
                )
            else:
                conversation = Conversation.objects.get(
                    id=conversation_id,
                    user=request.user
                )
            
            # Create message
            message = Message.objects.create(
                conversation=conversation,
                role=role,
                content=content,
                message_type=message_type,
                metadata=metadata
            )
            
            # Auto-generate title from first user message
            if not conversation.title and role == 'user':
                title = content[:50]
                if len(content) > 50:
                    title += "..."
                conversation.title = title
                conversation.save()
            
            return JsonResponse({
                'success': True,
                'message_id': str(message.id),
                'conversation_id': str(conversation.id),
                'conversation_title': conversation.get_title()
            })
            
        except Conversation.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Conversation not found'
            }, status=404)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
