# detector/urls.py
from django.urls import path
from .views import FakeNewsDetectorAPIView, RecentAnalysesAPIView, TrendDataAPIView, AnalysisDetailAPIView, EnhancedVerificationAPIView

urlpatterns = [
    path('detect/', FakeNewsDetectorAPIView.as_view(), name='detect_fake_news'),
    path('enhanced-verify/', EnhancedVerificationAPIView.as_view(), name='enhanced_verification'),
    path('recent-analyses/', RecentAnalysesAPIView.as_view(), name='recent_analyses'),
    path('analysis/<int:analysis_id>/', AnalysisDetailAPIView.as_view(), name='analysis_detail'),
    path('trends/volume/', TrendDataAPIView.as_view(), name='trends_volume'),
]
