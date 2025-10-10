from django.conf import settings
from django.db.models.signals import post_migrate
from django.dispatch import receiver


def _ensure_google_social_app():
    try:
        from allauth.socialaccount.models import SocialApp
        from django.contrib.sites.models import Site
    except Exception as e:
        print(f"[OAuth] Skipping SocialApp setup (imports failed): {e}")
        return

    client_id = getattr(settings, 'GOOGLE_CLIENT_ID', None)
    secret = getattr(settings, 'GOOGLE_CLIENT_SECRET', None)

    if not client_id or not secret:
        print("[OAuth] GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET not configured. Skipping SocialApp creation.")
        return

    try:
        app, created = SocialApp.objects.get_or_create(
            provider='google',
            defaults={
                'name': 'Google',
                'client_id': client_id,
                'secret': secret,
            },
        )
        if not created:
            # Update credentials if changed
            changed = False
            if app.client_id != client_id:
                app.client_id = client_id
                changed = True
            if app.secret != secret:
                app.secret = secret
                changed = True
            if changed:
                app.save()

        # Attach current site
        try:
            current_site = Site.objects.get(id=settings.SITE_ID)
        except Site.DoesNotExist:
            current_site = Site.objects.create(id=settings.SITE_ID, domain='127.0.0.1', name='Local')
        if current_site not in app.sites.all():
            app.sites.add(current_site)

        print(f"[OAuth] Google SocialApp {'created' if created else 'updated'} and linked to Site(id={settings.SITE_ID}).")
    except Exception as e:
        print(f"[OAuth] Error ensuring Google SocialApp: {e}")


@receiver(post_migrate)
def create_or_update_google_social_app(sender, **kwargs):
    # Run once after migrations for any app
    _ensure_google_social_app()
