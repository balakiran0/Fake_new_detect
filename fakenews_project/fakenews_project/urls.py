# fakenews_project/urls.py
from django.http import HttpResponse
from django.urls import path, include
from django.contrib import admin
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


def home(request):
    # Show landing page for unauthenticated users, redirect to dashboard if authenticated
    if request.user.is_authenticated:
        return redirect('dashboard:home')
    else:
        return render(request, "landing.html")


@login_required
def analysis_result(request):
    # Render the analysis results page (now protected)
    return render(request, "analysis_result.html")


urlpatterns = [
    path("", home),  # root URL
    path("analysis-result/", analysis_result, name="analysis_result"),
    path("admin/", admin.site.urls),
    path("api/", include("detector.urls")),
    path("dashboard/", include("dashboard.urls")),
    path("accounts/", include("allauth.urls")),
]
