# GCP Enterprise Interactions testing prerequisites

Review date: 2026-07-14

This document records what must be provisioned before the Gemini Enterprise
Interactions live-test goal can be completed. The local `GEMINI_API_KEY` proves
the Developer API path only; it does not authenticate the Enterprise path.

## Information and resources to obtain from GCP

### 1. Dedicated test project

Obtain the project ID and project number for a dedicated GCP test project. The
project must have:

- billing enabled;
- the Vertex AI API (`aiplatform.googleapis.com`) enabled;
- sufficient Gemini and Vertex AI quota; and
- organization policies that permit Vertex AI generative AI and the
  Interactions operations under test.

The human or provisioning automation that enables services may need elevated
Service Usage and IAM permissions. Those administrative permissions must not be
granted to the runtime service account.

Reference: [Vertex AI generative AI quickstart](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart).

### 2. Audited model and location

Obtain an exact bare Enterprise model ID and a location in which that model is
available to this project. Do not infer Enterprise availability from a model's
Developer API availability. Record both values because successful evidence is
valid only for the tested project/location/model/API-version tuple.

Required values:

```text
GEMINI_ENTERPRISE_PROJECT=<project-id>
GEMINI_ENTERPRISE_LOCATION=<location>
GEMINI_ENTERPRISE_MODEL=<bare-model-id>
```

### 3. Dedicated keyless runtime service account

Create a service account used only by the Enterprise live smoke and obtain its
email address. Grant `roles/aiplatform.user` on the test project, or a tighter
custom role whose exact endpoint permissions have been verified. Do not grant
Owner, Editor, Service Usage Admin, or project-wide IAM administration to this
account.

Required value:

```text
GCP_SERVICE_ACCOUNT=<service-account>@<project-id>.iam.gserviceaccount.com
```

Do not create or share a service-account JSON key. GitHub CI authenticates with
short-lived credentials through Workload Identity Federation.

### 4. GitHub OIDC Workload Identity Federation provider

Create a Workload Identity Pool and GitHub OIDC provider with issuer:

```text
https://token.actions.githubusercontent.com/
```

Obtain the provider's full resource name:

```text
projects/<project-number>/locations/global/workloadIdentityPools/<pool-id>/providers/<provider-id>
```

Map at least `google.subject` and attributes for repository ID, repository owner
ID, ref, and environment. Restrict the provider condition to this immutable
identity and deployment context:

| Claim | Required value |
|---|---|
| `repository_id` | `1262464913` |
| `repository_owner_id` | `94211695` |
| `ref` | `refs/heads/master` |
| `environment` | `gemini-live-enterprise` |

Use the provider's default audience. Grant the restricted federated principal
`roles/iam.workloadIdentityUser` on the dedicated runtime service account so it
can impersonate that account. No static credential is required.

Required value:

```text
GCP_WORKLOAD_IDENTITY_PROVIDER=projects/<project-number>/locations/global/workloadIdentityPools/<pool-id>/providers/<provider-id>
```

References:

- [Google Cloud WIF for deployment pipelines](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines)
- [Google Cloud WIF best practices](https://docs.cloud.google.com/iam/docs/best-practices-for-using-workload-identity-federation)
- [GitHub OIDC claims](https://docs.github.com/en/actions/reference/security/oidc)
- [`google-github-actions/auth`](https://github.com/google-github-actions/auth)

## GitHub configuration also required

This portion is not obtained from GCP, but it is required for goal completion:

1. Add a second trusted GitHub user or team capable of reviewing deployments.
2. Create the protected `gemini-live-enterprise` environment.
3. Require that reviewer, prevent self-review/bypass, and restrict deployment to
   the default branch.
4. Add these environment variables:

   ```text
   GCP_WORKLOAD_IDENTITY_PROVIDER
   GCP_SERVICE_ACCOUNT
   GEMINI_ENTERPRISE_PROJECT
   GEMINI_ENTERPRISE_LOCATION
   ```

The audited model and API version are explicit workflow-dispatch inputs rather
than stored environment values.

## Local and CI environment-variable inventory

The committed [`.env.example`](../.env.example) is the canonical placeholder
inventory. It intentionally contains no credential values.

| Variable | Purpose | Secret? |
|---|---|---|
| `RUN_GEMINI_LIVE_TESTS` | Explicitly enable Developer live calls | No |
| `GEMINI_API_KEY` | Ephemeral Developer API credential | Yes |
| `GEMINI_LIVE_MODEL` | Developer smoke model override | No |
| `RUN_GEMINI_ENTERPRISE_LIVE_TESTS` | Explicitly enable Enterprise live calls | No |
| `GEMINI_ENTERPRISE_PROJECT` | Enterprise GCP project ID | Treat as sensitive identifier |
| `GEMINI_ENTERPRISE_LOCATION` | Enterprise Vertex AI location | No |
| `GEMINI_ENTERPRISE_MODEL` | Audited bare Enterprise model ID | No |
| `GEMINI_ENTERPRISE_API_VERSION` | `v1beta1` or `v1` probe selection | No |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full WIF provider resource name | Treat as sensitive identifier |
| `GCP_SERVICE_ACCOUNT` | Runtime service-account email | Treat as sensitive identifier |

The ephemeral `GEMINI_API_KEY` currently present in the terminal must remain in
the process environment or an ignored local `.env`; its value must never be
copied into `.env.example`, reports, logs, commits, or GitHub variables. In the
Developer workflow it belongs in the protected environment's
`GEMINI_API_KEY` GitHub secret.

For a local Enterprise run, authenticate with Google Application Default
Credentials outside the repository. Do not place an ADC file or its contents in
this repository. The GitHub workflow obtains short-lived ADC automatically from
the WIF provider.

## Evidence required to finish the goal

After provisioning, dispatch
`gemini-enterprise-live-smoke.yml` twice from `master` using the same audited
model: once with `v1beta1` and once with `v1`. At least one run must succeed,
and neither run may skip any of the three tests. Retain the sanitized run URLs
and results for unary generation, streaming cleanup, and stored semantic
continuation/cleanup. Runtime success alone does not establish pricing or every
catalog capability; those fields still require authoritative service-specific
evidence before the Enterprise model can be promoted.
