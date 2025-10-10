from django import forms
from django.contrib.auth.forms import UserCreationForm, PasswordResetForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from allauth.account.forms import SignupForm
from django.contrib.auth import get_user_model


class CustomSignUpForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address'
        })
    )
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name'
        })
    )

    class Meta:
        model = User
        fields = ('first_name', 'email', 'username', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Customize field widgets and labels
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Choose a username'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Create a strong password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })
        
        # Update labels
        self.fields['first_name'].label = 'Name'
        self.fields['email'].label = 'Email'
        self.fields['username'].label = 'Username'
        self.fields['password1'].label = 'Password'
        self.fields['password2'].label = 'Confirm Password'

    def clean_email(self):
        email = self.cleaned_data.get('email')
        existing_user = User.objects.filter(email=email).first()
        if existing_user:
            # If user exists but is not active (not verified), delete the old account
            if not existing_user.is_active:
                existing_user.delete()
                return email
            else:
                raise ValidationError("A user with this email already exists and is verified.")
        return email
    
    def clean_username(self):
        username = self.cleaned_data.get('username')
        existing_user = User.objects.filter(username=username).first()
        if existing_user:
            # If user exists but is not active (not verified), delete the old account
            if not existing_user.is_active:
                existing_user.delete()
                return username
            else:
                raise ValidationError("A user with that username already exists and is verified.")
        return username
    
    def clean(self):
        # Clean unverified accounts before running parent validation
        email = self.data.get('email')
        username = self.data.get('username')
        
        if email:
            existing_user = User.objects.filter(email=email, is_active=False).first()
            if existing_user:
                existing_user.delete()
        
        if username:
            existing_user = User.objects.filter(username=username, is_active=False).first()
            if existing_user:
                existing_user.delete()
        
        return super().clean()

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.is_active = False  # User needs to confirm email first
        if commit:
            user.save()
        return user


class CustomAllauthSignupForm(SignupForm):
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Customize field widgets and labels
        self.fields['email'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your email address'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Create a strong password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })
        
        # Update labels
        self.fields['first_name'].label = 'Name'
        self.fields['email'].label = 'Email'
        self.fields['password1'].label = 'Password'
        self.fields['password2'].label = 'Confirm Password'

    def save(self, request):
        user = super().save(request)
        user.first_name = self.cleaned_data['first_name']
        user.save()
        return user


class CustomPasswordResetForm(PasswordResetForm):
    """
    Custom password reset form with validation for non-existent accounts
    """
    email = forms.EmailField(
        max_length=254,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'autocomplete': 'email'
        })
    )

    def clean_email(self):
        email = self.cleaned_data.get('email')
        
        if not email:
            raise ValidationError("Please enter your email address.")
        
        # Check if email format is valid
        try:
            forms.EmailField().clean(email)
        except ValidationError:
            raise ValidationError("Please enter a valid email address.")
        
        # Check if user exists with this email
        User = get_user_model()
        user_exists = User.objects.filter(email=email).exists()
        
        if not user_exists:
            raise ValidationError(
                "No account found with this email address. "
                "Please check your email or create a new account."
            )
        
        # Check if user account is active
        user = User.objects.filter(email=email).first()
        if user and not user.is_active:
            raise ValidationError(
                "Your account is not activated yet. "
                "Please check your email for the activation link or contact support."
            )
        
        return email

    def get_users(self, email):
        """
        Override to ensure we only return active users
        """
        User = get_user_model()
        active_users = User.objects.filter(
            email__iexact=email,
            is_active=True
        )
        return (
            u for u in active_users
            if u.has_usable_password() and
            hasattr(u, 'email') and u.email
        )
