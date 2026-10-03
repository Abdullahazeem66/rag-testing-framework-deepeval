# Account Security

## Two-factor authentication

Orbitly supports two-factor authentication (2FA) using an authenticator app (such as Google Authenticator or 1Password) or SMS codes. SMS codes are not available in every country; an authenticator app is the recommended method. Workspace admins on Pro and above can require 2FA for all members.

## Single sign-on (SSO)

SAML 2.0 single sign-on is available on the Business and Enterprise plans. Supported identity providers include Okta, Microsoft Entra ID (Azure AD) and Google Workspace. SCIM user provisioning is available on Enterprise only.

## Passwords

Passwords must be at least 12 characters long. Orbitly staff will never ask for your password by email, chat or phone. To reset a password, use the "Forgot password" link on the sign-in page; the reset link expires after 1 hour.

## Sessions and lockout

By default, sessions stay signed in for 30 days. Admins on Business and Enterprise can set a session timeout anywhere between 1 hour and 90 days. After 10 consecutive failed sign-in attempts, the account is locked for 15 minutes.

## Reporting a vulnerability

Security issues can be reported to security@orbitly.example. Orbitly runs a responsible disclosure program and responds to reports within 2 business days.
