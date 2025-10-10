from django.urls import path
from django.urls import reverse_lazy
from django.contrib.auth.views import LogoutView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
from .views import (
    DashboardLoginView, DashboardHomeView, SignUpView, confirm_email, 
    VerifyOTPView, ResendOTPView, CustomPasswordResetView, CustomPasswordResetDoneView, 
    ProfileView, CustomLogoutView, ConversationListView, ConversationDetailView,
    CreateConversationView, DeleteConversationView, SaveMessageView
)

app_name = "dashboard"

urlpatterns = [
    path("", DashboardHomeView.as_view(), name="home"),
    path("login/", DashboardLoginView.as_view(), name="login"),
    path("signup/", SignUpView.as_view(), name="signup"),
    path("logout/", CustomLogoutView.as_view(), name="logout"),
    path("profile/", ProfileView.as_view(), name="profile"),
    path("verify-otp/", VerifyOTPView.as_view(), name="verify_otp"),
    path("resend-otp/", ResendOTPView.as_view(), name="resend_otp"),
    path("confirm-email/<uuid:token>/", confirm_email, name="confirm_email"),
    
    # Password Reset URLs
    path("password-reset/", CustomPasswordResetView.as_view(), name="password_reset"),
    path("password-reset/done/", CustomPasswordResetDoneView.as_view(), name="password_reset_done"),
    path("password-reset-confirm/<uidb64>/<token>/", PasswordResetConfirmView.as_view(
        template_name='registration/password_reset_confirm.html',
        success_url=reverse_lazy('dashboard:password_reset_complete')
    ), name="password_reset_confirm"),
    path("password-reset-complete/", PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'), name="password_reset_complete"),
    
    # Conversation Management APIs
    path("api/conversations/", ConversationListView.as_view(), name="conversation_list"),
    path("api/conversations/create/", CreateConversationView.as_view(), name="create_conversation"),
    path("api/conversations/<uuid:conversation_id>/", ConversationDetailView.as_view(), name="conversation_detail"),
    path("api/conversations/<uuid:conversation_id>/delete/", DeleteConversationView.as_view(), name="delete_conversation"),
    path("api/messages/save/", SaveMessageView.as_view(), name="save_message"),
]
