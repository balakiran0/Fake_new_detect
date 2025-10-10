# Template Cleanup Plan

## 🚨 Current Issues
- **22 HTML templates** with inconsistent layouts
- **No base template** (until now)
- **Multiple duplicate templates**
- **Different design systems** across pages

## 📋 Template Analysis

### ✅ Keep & Refactor
1. **`templates/index.html`** → Convert to extend `layouts/dashboard.html`
2. **`templates/dashboard/login.html`** → Convert to extend `layouts/auth.html`
3. **`templates/dashboard/signup.html`** → Convert to extend `layouts/auth.html`
4. **`templates/dashboard/profile.html`** → Convert to extend `layouts/dashboard.html`

### 🔄 Consolidate Duplicates
**Password Reset Templates** (Choose one set, delete others):
- `account/password_reset.html` ✅ KEEP
- `account/password_reset_done.html` ✅ KEEP  
- `account/password_reset_confirm.html` ✅ KEEP
- `account/password_reset_complete.html` ✅ KEEP
- ~~`registration/password_reset_*.html`~~ ❌ DELETE (duplicates)

### 🗑️ Delete Unused/Redundant
- `templates/landing.html` (if not used)
- `templates/analysis_result.html` (if replaced by dashboard)
- `templates/dashboard/verify_otp.html` (if not needed)
- All `registration/` templates (use `account/` versions)

### 📧 Email Templates (Keep as-is)
- `dashboard/emails/confirmation_email.html`
- `dashboard/emails/otp_email.html`
- `account/email/email_confirmation_message.html`

## 🎯 Refactoring Strategy

### 1. Create Base Templates ✅ DONE
- `templates/base.html` - Master template
- `templates/layouts/auth.html` - For login/signup pages
- `templates/layouts/dashboard.html` - For main app pages

### 2. Convert Existing Templates
Each template should:
```html
{% extends "layouts/auth.html" %}  <!-- or dashboard.html -->

{% block title %}Page Title{% endblock %}

{% block form_content %}  <!-- or main_content -->
<!-- Page specific content -->
{% endblock %}
```

### 3. Unified Design System
- **Colors**: Consistent CSS variables
- **Typography**: Inter font family
- **Components**: Shared button styles, forms, etc.
- **Responsive**: Mobile-first approach

## 📊 Before vs After

### Before:
- 22 templates
- ~15,000 lines of duplicated CSS
- 5+ different color schemes
- Inconsistent UX

### After:
- 3 base templates + 8-10 content templates
- ~2,000 lines of unified CSS
- 1 consistent design system
- Unified UX

## 🚀 Implementation Steps

1. ✅ Create base templates (DONE)
2. 🔄 Convert main templates one by one
3. 🗑️ Delete duplicate/unused templates
4. 🧪 Test all pages work correctly
5. 📝 Update documentation

## 🎨 Design Benefits

- **Maintainability**: Change CSS in one place
- **Consistency**: Same look and feel everywhere  
- **Performance**: Less CSS to load
- **Developer Experience**: Easier to add new pages
- **User Experience**: Consistent navigation and styling

## 🔧 Next Actions

1. Convert `dashboard/login.html` to use `layouts/auth.html`
2. Convert `index.html` to use `layouts/dashboard.html`
3. Delete duplicate password reset templates
4. Test authentication flow
5. Clean up unused templates
