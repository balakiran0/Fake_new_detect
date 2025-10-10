#!/usr/bin/env python
"""
Script to fix the Django site domain for password reset emails
Run this script to update the site domain from example.com to localhost:8000
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

from django.contrib.sites.models import Site

def fix_site_domain():
    """Update the site domain to localhost:8000"""
    try:
        # Get the default site (ID=1)
        site = Site.objects.get(pk=1)
        
        print(f"Current site configuration:")
        print(f"  Domain: {site.domain}")
        print(f"  Name: {site.name}")
        
        # Update the site
        site.domain = '127.0.0.1:8000'
        site.name = 'Fake News Detector'
        site.save()
        
        print(f"\n✅ Site updated successfully!")
        print(f"  New Domain: {site.domain}")
        print(f"  New Name: {site.name}")
        
        print(f"\n🔗 Password reset links will now use: http://{site.domain}")
        
    except Site.DoesNotExist:
        print("❌ Default site not found. Creating new site...")
        site = Site.objects.create(
            pk=1,
            domain='127.0.0.1:8000',
            name='Fake News Detector'
        )
        print(f"✅ Site created successfully!")
        print(f"  Domain: {site.domain}")
        print(f"  Name: {site.name}")
    
    except Exception as e:
        print(f"❌ Error updating site: {str(e)}")

if __name__ == "__main__":
    print("🔧 Fixing Django site domain for password reset emails...")
    print("=" * 60)
    fix_site_domain()
    print("=" * 60)
    print("✅ Done! Try sending a password reset email now.")
