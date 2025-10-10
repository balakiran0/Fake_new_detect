from django.core.management.base import BaseCommand
from django.contrib.sites.models import Site


class Command(BaseCommand):
    help = 'Fix the Django site domain for password reset emails'

    def handle(self, *args, **options):
        try:
            # Get the default site (ID=1)
            site = Site.objects.get(pk=1)
            
            self.stdout.write(f"Current site configuration:")
            self.stdout.write(f"  Domain: {site.domain}")
            self.stdout.write(f"  Name: {site.name}")
            
            # Update the site
            site.domain = '127.0.0.1:8000'
            site.name = 'Fake News Detector'
            site.save()
            
            self.stdout.write(
                self.style.SUCCESS(f"✅ Site updated successfully!")
            )
            self.stdout.write(f"  New Domain: {site.domain}")
            self.stdout.write(f"  New Name: {site.name}")
            
            self.stdout.write(f"🔗 Password reset links will now use: http://{site.domain}")
            
        except Site.DoesNotExist:
            self.stdout.write("❌ Default site not found. Creating new site...")
            site = Site.objects.create(
                pk=1,
                domain='127.0.0.1:8000',
                name='Fake News Detector'
            )
            self.stdout.write(
                self.style.SUCCESS(f"✅ Site created successfully!")
            )
            self.stdout.write(f"  Domain: {site.domain}")
            self.stdout.write(f"  Name: {site.name}")
        
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Error updating site: {str(e)}")
            )
