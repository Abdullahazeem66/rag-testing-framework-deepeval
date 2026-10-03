# API and Integrations

## REST API

Orbitly offers a REST API (version 2) at https://api.orbitly.example/v2. Requests are authenticated with a personal access token sent in the Authorization header as a Bearer token. Personal access tokens expire after 90 days and can be revoked at any time from Settings > Developer.

## Rate limits

API rate limits depend on the plan: Free allows 60 requests per minute, Pro allows 300 requests per minute, Business allows 1,000 requests per minute and Enterprise limits are custom. When the limit is exceeded, the API returns HTTP 429 with a Retry-After header.

## Webhooks

Webhooks send an HTTP POST to your endpoint when tasks, projects or comments change. Your endpoint must respond with a 2xx status within 10 seconds. Failed deliveries are retried up to 5 times with exponential backoff over 24 hours, after which the webhook is disabled and the workspace admins are emailed.

## Native integrations

Orbitly integrates natively with Slack, Microsoft Teams, GitHub, GitLab, Google Drive and Zapier. The GitHub and GitLab integrations link commits and pull requests to tasks when the task ID (for example ORB-123) appears in the commit message or branch name. Jira import is available as a one-time migration tool, not as a live sync.
