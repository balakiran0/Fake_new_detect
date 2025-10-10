from django.apps import AppConfig


class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'

    def ready(self):
        # Import signals to auto-create/attach Google SocialApp after migrations
        try:
            from . import signals  # noqa: F401
            # Also ensure once at startup (in case no migrations run during this boot)
            try:
                signals._ensure_google_social_app()
            except Exception as e:
                print(f"[OAuth] Ensure SocialApp on startup failed: {e}")
        except Exception as e:
            # Do not crash app startup due to optional social setup
            print(f"[OAuth] Signals not loaded: {e}")
