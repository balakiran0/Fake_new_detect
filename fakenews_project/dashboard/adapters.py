from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.account.models import EmailAddress
from django.contrib.auth import get_user_model


class GoogleSocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Force-mark email as verified and primary for social (Google) signups
    so users are logged in immediately without a confirmation email.
    """
    def pre_social_login(self, request, sociallogin):
        """
        If a user with the same email already exists (from normal signup),
        connect this Google social account to that user and proceed.
        This prevents the common 'Third-Party Login Failure' due to duplicate email.
        """
        try:
            if sociallogin.is_existing:
                return
            email = None
            if sociallogin.account and sociallogin.account.extra_data:
                email = sociallogin.account.extra_data.get('email')
            if not email:
                # Fallback to user object email if present
                email = getattr(sociallogin.user, 'email', None)
            if not email:
                return
            User = get_user_model()
            try:
                existing = User.objects.get(email__iexact=email)
            except User.DoesNotExist:
                return
            # Link the social account to the existing user
            sociallogin.connect(request, existing)
        except Exception:
            # Do not block login on adapter errors; let allauth handle gracefully
            return

    def save_user(self, request, sociallogin, form=None):
        user = super().save_user(request, sociallogin, form)

        email = user.email or (sociallogin.account.extra_data.get('email') if sociallogin and sociallogin.account else None)
        if email:
            # Ensure EmailAddress exists and is verified+primary
            email_obj, created = EmailAddress.objects.get_or_create(
                user=user,
                email=email,
                defaults={
                    'verified': True,
                    'primary': True,
                }
            )
            changed = False
            if not email_obj.verified:
                email_obj.verified = True
                changed = True
            if not email_obj.primary:
                email_obj.primary = True
                changed = True
            if changed:
                email_obj.save()
        return user
