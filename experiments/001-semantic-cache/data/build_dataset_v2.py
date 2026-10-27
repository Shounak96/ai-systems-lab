"""Build the reviewed V2 benchmark from manually authored query families.

The script performs no random text generation. It only materializes and validates
the curated family definitions below. Split assignment uses a fixed seed and
keeps every family wholly in development or held-out test.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random

RANDOM_SEED = 20261023
OUTPUT = Path(__file__).with_name("test_cases_v2.csv")
EXACT_REPEAT_FAMILIES = {
    "backup_policy", "rate_limit", "tls_policy", "service_status",
    "postgres_upgrade", "maintenance_date", "feature_flag", "mfa",
    "disaster_recovery", "cache_invalidation", "tenant_balance", "http_method",
}


@dataclass(frozen=True)
class Variant:
    query: str
    eligible: bool
    category: str
    explanation: str
    query_scope: str = "global"
    query_user: str = ""


@dataclass(frozen=True)
class Family:
    family_id: str
    seed_query: str
    variants: tuple[Variant, ...]
    seed_scope: str = "global"
    seed_user: str = ""


def p(query: str, category: str, explanation: str) -> Variant:
    return Variant(query, True, category, explanation)


def n(query: str, category: str, explanation: str) -> Variant:
    return Variant(query, False, category, explanation)


FAMILIES = (
    Family("backup_policy", "Should automatic database backups be enabled in production?", (
        p("In production, should we turn on automatic database backups?", "positive_paraphrase", "Same environment and backup requirement."),
        p("Are automated production database backups recommended?", "positive_synonym", "Automated and automatic express the same requirement."),
        p("Should production databases have automatic backups enabled?", "positive_word_order", "Word order changes without changing policy."),
        n("Should automatic database backups not be enabled in production?", "negative_negation", "Explicit negation reverses the requested policy."),
        n("Should automatic database backups be enabled in staging?", "negative_condition", "The deployment environment changes."),
        n("Should manual database backups be enabled in production?", "negative_operational", "The backup mechanism changes from automatic to manual."),
    )),
    Family("rate_limit", "Does API version 2 allow 100 requests per minute?", (
        p("Can API v2 accept 100 requests each minute?", "positive_technical", "Same API version, quantity, and interval."),
        p("Is the per-minute limit for API v2 equal to 100 requests?", "positive_word_order", "Same limit expressed with different order."),
        p("For version 2.0 of the API, are 100 requests allowed per minute?", "positive_version_format", "Version 2 and 2.0 denote the same major version in this fixture."),
        n("Does API version 2 allow 200 requests per minute?", "negative_quantity", "The permitted request count changes."),
        n("Does API version 3 allow 100 requests per minute?", "negative_version", "The API version changes."),
        n("Does API version 2 deny 100 requests per minute?", "negative_permission", "Allow is replaced by deny."),
    )),
    Family("replicas", "Deploy the checkout service with 3 replicas in production.", (
        p("In production, run 3 replicas of the checkout service.", "positive_word_order", "Same service, environment, and replica count."),
        p("Configure production checkout to use 3 service replicas.", "positive_technical", "Equivalent deployment instruction."),
        p("Set the checkout service replica count to 3 for production.", "positive_paraphrase", "Same desired state."),
        n("Deploy the checkout service with 5 replicas in production.", "negative_quantity", "Replica count changes."),
        n("Deploy the payment service with 3 replicas in production.", "negative_entity", "Target service changes."),
        n("Deploy the checkout service with 3 replicas in development.", "negative_condition", "Target environment changes."),
    )),
    Family("tls_policy", "Is TLS 1.3 required after January 15, 2026?", (
        p("After January 15, 2026, must clients use TLS 1.3?", "positive_word_order", "Same date, protocol version, and requirement."),
        p("Do we require TLS version 1.3 starting January 15, 2026?", "positive_technical", "Equivalent policy wording."),
        p("Beginning January 15 2026, is TLS v1.3 mandatory?", "positive_date_format", "Same calendar date and version."),
        n("Is TLS 1.2 required after January 15, 2026?", "negative_version", "Protocol version changes."),
        n("Is TLS 1.3 required after January 16, 2026?", "negative_date", "Effective date changes."),
        n("Is TLS 1.3 optional after January 15, 2026?", "negative_permission", "Required changes to optional."),
    )),
    Family("document_access", "May viewers download confidential reports?", (
        p("Are users with the viewer role allowed to download confidential reports?", "positive_paraphrase", "Same role, action, and resource."),
        p("Can a viewer download a confidential report?", "positive_synonym", "May and can express the same permission question."),
        p("For confidential reports, is downloading permitted to viewers?", "positive_word_order", "Same permission with reordered wording."),
        n("May editors download confidential reports?", "negative_permission", "The principal role changes."),
        n("May viewers delete confidential reports?", "negative_permission", "The requested action changes."),
        n("May viewers not download confidential reports?", "negative_negation", "Negation reverses the permission question."),
    )),
    Family("service_status", "Is the payments API available in the US region?", (
        p("Is the US-region payments API online?", "positive_synonym", "Available and online are equivalent status terms here."),
        p("In the US region, is the payments API up?", "positive_word_order", "Same service, region, and status."),
        p("Can clients currently reach the payments API in the US region?", "positive_paraphrase", "Reachability asks the same availability question."),
        n("Is the payments API unavailable in the US region?", "negative_operational", "Availability state is reversed."),
        n("Is the payments API available in the EU region?", "negative_condition", "Region changes."),
        n("Is the orders API available in the US region?", "negative_entity", "Service identity changes."),
    )),
    Family("postgres_upgrade", "Can PostgreSQL 15 use this migration without downtime?", (
        p("Does this migration run on PostgreSQL v15 with no downtime?", "positive_technical", "Same database major version and downtime condition."),
        p("For PostgreSQL version 15, is the migration zero-downtime?", "positive_synonym", "Zero-downtime is equivalent to without downtime."),
        p("Is this a no-downtime migration for PostgreSQL 15?", "positive_word_order", "Same compatibility question."),
        n("Can PostgreSQL 14 use this migration without downtime?", "negative_version", "Database major version changes."),
        n("Can MySQL 15 use this migration without downtime?", "negative_entity", "Database product changes."),
        n("Can PostgreSQL 15 use this migration with downtime?", "negative_operational", "Downtime condition reverses."),
    )),
    Family("redis_ttl", "Set the Redis session TTL to 30 minutes.", (
        p("Configure Redis sessions to expire after 30 minutes.", "positive_technical", "TTL and expiry duration are equivalent."),
        p("Use a 30-minute expiration for sessions in Redis.", "positive_word_order", "Same product, object, and duration."),
        p("For Redis, make each session live for 30 minutes.", "positive_paraphrase", "Same lifetime requirement."),
        n("Set the Redis session TTL to 60 minutes.", "negative_quantity", "Expiration duration changes."),
        n("Set the Redis cache-entry TTL to 30 minutes.", "negative_entity", "The cached object changes."),
        n("Set the MongoDB session TTL to 30 minutes.", "negative_entity", "Storage product changes."),
    )),
    Family("kubernetes_namespace", "Scale the billing deployment to 4 pods in Kubernetes namespace prod.", (
        p("In Kubernetes namespace prod, run 4 billing pods.", "positive_word_order", "Same platform, namespace, workload, and count."),
        p("Set billing to 4 pods in the prod Kubernetes namespace.", "positive_technical", "Equivalent scaling instruction."),
        p("Configure four pods for billing in Kubernetes namespace prod.", "positive_number_word", "Four and 4 express the same count; this challenges a lexical number guard."),
        n("Scale the billing deployment to 5 pods in Kubernetes namespace prod.", "negative_quantity", "Pod count changes."),
        n("Scale the billing deployment to 4 pods in Kubernetes namespace staging.", "negative_condition", "Namespace changes."),
        n("Scale the orders deployment to 4 pods in Kubernetes namespace prod.", "negative_entity", "Workload changes."),
    )),
    Family("github_branch", "Allow GitHub maintainers to merge into the main branch.", (
        p("May maintainers merge pull requests into main on GitHub?", "positive_paraphrase", "Same role, platform, branch, and action."),
        p("On GitHub, permit the maintainer role to merge to main.", "positive_word_order", "Equivalent access rule."),
        p("Can repository maintainers merge changes into GitHub's main branch?", "positive_technical", "Same permission with explicit repository context."),
        n("Allow GitHub contributors to merge into the main branch.", "negative_permission", "Role changes."),
        n("Deny GitHub maintainers permission to merge into the main branch.", "negative_permission", "Permission polarity reverses."),
        n("Allow GitLab maintainers to merge into the main branch.", "negative_entity", "Hosting product changes."),
    )),
    Family("s3_region", "Create the audit bucket in Amazon S3 region us-east-1.", (
        p("In Amazon S3, create the audit bucket in us-east-1.", "positive_word_order", "Same provider, bucket, and region."),
        p("Provision an audit S3 bucket in us-east-1.", "positive_synonym", "Provision and create are equivalent here."),
        p("Use us-east-1 for the Amazon S3 audit bucket.", "positive_technical", "Same placement requirement."),
        n("Create the audit bucket in Amazon S3 region us-west-2.", "negative_condition", "Region changes."),
        n("Create the logs bucket in Amazon S3 region us-east-1.", "negative_entity", "Bucket identity changes."),
        n("Create the audit bucket in Azure region us-east-1.", "negative_entity", "Cloud provider changes."),
    )),
    Family("python_runtime", "Does the worker support Python 3.12?", (
        p("Can the worker run on Python version 3.12?", "positive_technical", "Same runtime version compatibility."),
        p("Is Python v3.12 supported by the worker?", "positive_word_order", "Same support question."),
        p("Will the worker operate with Python 3.12?", "positive_synonym", "Operate with and support are equivalent here."),
        n("Does the worker support Python 3.11?", "negative_version", "Runtime version changes."),
        n("Does the worker support Java 3.12?", "negative_entity", "Runtime product changes."),
        n("Does the worker not support Python 3.12?", "negative_negation", "Negation reverses compatibility."),
    )),
    Family("node_runtime", "Install Node.js 22 for the build agent.", (
        p("Use Node.js version 22 on the build agent.", "positive_technical", "Same runtime version and target."),
        p("Configure the build agent with Node.js v22.", "positive_word_order", "Equivalent installation request."),
        p("The build agent should run Node.js 22.", "positive_paraphrase", "Same desired runtime."),
        n("Install Node.js 20 for the build agent.", "negative_version", "Runtime major version changes."),
        n("Install Python 22 for the build agent.", "negative_entity", "Runtime product changes."),
        n("Remove Node.js 22 from the build agent.", "negative_operational", "Operation changes from install to remove."),
    )),
    Family("maintenance_date", "Schedule maintenance for March 10, 2027 at 02:00 UTC.", (
        p("At 02:00 UTC on March 10, 2027, schedule maintenance.", "positive_word_order", "Same date and time."),
        p("Plan the maintenance window for March 10 2027 at 02:00 UTC.", "positive_date_format", "Punctuation changes but timestamp is the same."),
        p("Book system maintenance on March 10, 2027 at 02:00 UTC.", "positive_synonym", "Book and schedule are equivalent."),
        n("Schedule maintenance for March 11, 2027 at 02:00 UTC.", "negative_date", "Calendar day changes."),
        n("Schedule maintenance for March 10, 2027 at 03:00 UTC.", "negative_date", "Time changes."),
        n("Cancel maintenance for March 10, 2027 at 02:00 UTC.", "negative_operational", "Operation changes from schedule to cancel."),
    )),
    Family("retention", "Retain application logs for 90 days.", (
        p("Keep application logs for 90 days.", "positive_synonym", "Keep and retain express the same policy."),
        p("Use a 90-day retention period for application logs.", "positive_technical", "Same duration and data type."),
        p("Application log retention should be 90 days.", "positive_word_order", "Same policy with reordered phrasing."),
        n("Retain application logs for 30 days.", "negative_quantity", "Retention duration changes."),
        n("Retain audit logs for 90 days.", "negative_entity", "Log type changes."),
        n("Delete application logs immediately instead of retaining them for 90 days.", "negative_operational", "Retention behavior reverses."),
    )),
    Family("pagination", "Return 50 records per API page.", (
        p("Set the API page size to 50 records.", "positive_technical", "Same pagination size."),
        p("Each API page should contain 50 records.", "positive_word_order", "Same result size."),
        p("Paginate API results in batches of 50 records.", "positive_synonym", "Batch size and page size are equivalent here."),
        n("Return 100 records per API page.", "negative_quantity", "Page size changes."),
        n("Return 50 records for the entire API response.", "negative_condition", "Limit applies to the full response, not each page."),
        n("Return 50 users per API page instead of records.", "negative_entity", "Resource type changes."),
    )),
    Family("retry_policy", "Retry failed requests 3 times with backoff enabled.", (
        p("With backoff enabled, make 3 retries for failed requests.", "positive_word_order", "Same attempts and backoff policy."),
        p("Attempt failed requests again 3 times using backoff.", "positive_synonym", "Equivalent retry instruction."),
        p("Use three retry attempts and enable backoff for failed requests.", "positive_number_word", "Three and 3 express the same count."),
        n("Retry failed requests 5 times with backoff enabled.", "negative_quantity", "Retry count changes."),
        n("Retry failed requests 3 times with backoff disabled.", "negative_negation", "Backoff state changes."),
        n("Never retry failed requests; backoff is disabled.", "negative_operational", "Retry behavior is reversed."),
    )),
    Family("timeout", "Set the upstream request timeout to 30 seconds.", (
        p("Configure a 30-second timeout for upstream requests.", "positive_word_order", "Same duration and request class."),
        p("Upstream requests should time out after 30 seconds.", "positive_technical", "Equivalent timeout behavior."),
        p("Use 30 seconds as the upstream request timeout.", "positive_paraphrase", "Same configuration value."),
        n("Set the upstream request timeout to 60 seconds.", "negative_quantity", "Timeout duration changes."),
        n("Set the downstream request timeout to 30 seconds.", "negative_condition", "Traffic direction changes."),
        n("Disable the upstream request timeout of 30 seconds.", "negative_operational", "Timeout is disabled."),
    )),
    Family("feature_flag", "Enable the new checkout feature for tenant alpha.", (
        p("Turn on the new checkout feature for tenant alpha.", "positive_synonym", "Turn on and enable have the same effect."),
        p("For tenant alpha, enable the new checkout feature.", "positive_word_order", "Same tenant and flag state."),
        p("Tenant alpha should have the new checkout feature enabled.", "positive_paraphrase", "Same desired state."),
        n("Disable the new checkout feature for tenant alpha.", "negative_negation", "Flag state reverses."),
        n("Enable the new checkout feature for tenant beta.", "negative_tenant", "Tenant identity changes."),
        n("Enable the legacy checkout feature for tenant alpha.", "negative_entity", "Feature identity changes."),
    )),
    Family("mfa", "Require MFA for administrator sign-in.", (
        p("Administrators must use multi-factor authentication to sign in.", "positive_synonym", "MFA expansion preserves the policy."),
        p("For administrator login, require MFA.", "positive_word_order", "Same role and authentication rule."),
        p("Make multi-factor authentication mandatory for administrator sign-in.", "positive_paraphrase", "Mandatory and require are equivalent."),
        n("Do not require MFA for administrator sign-in.", "negative_negation", "Requirement is explicitly negated."),
        n("Require MFA for viewer sign-in.", "negative_permission", "Affected role changes."),
        n("Require a password but no MFA for administrator sign-in.", "negative_operational", "Authentication requirement changes."),
    )),
    Family("encryption", "Encrypt backups with AES-256 before upload.", (
        p("Before uploading backups, encrypt them using AES-256.", "positive_word_order", "Same algorithm and operation order."),
        p("Use AES-256 encryption on backups prior to upload.", "positive_technical", "Equivalent encryption requirement."),
        p("Backups must be AES-256 encrypted before they are uploaded.", "positive_paraphrase", "Same security control."),
        n("Encrypt backups with AES-128 before upload.", "negative_version", "Algorithm strength changes."),
        n("Upload backups without encryption.", "negative_negation", "Encryption requirement is removed."),
        n("Encrypt logs with AES-256 before upload.", "negative_entity", "Protected data type changes."),
    )),
    Family("logging", "Set production logging to warning level.", (
        p("Use warning-level logs in production.", "positive_word_order", "Same environment and log level."),
        p("Configure the production log level as warning.", "positive_technical", "Equivalent logging setting."),
        p("Production should emit warning-level logging.", "positive_paraphrase", "Same desired level."),
        n("Set production logging to debug level.", "negative_condition", "Log level changes."),
        n("Set staging logging to warning level.", "negative_condition", "Environment changes."),
        n("Disable production logging instead of using warning level.", "negative_operational", "Logging is disabled."),
    )),
    Family("disaster_recovery", "Fail over to the secondary region when the primary is unavailable.", (
        p("If the primary region is down, switch to the secondary region.", "positive_synonym", "Down and unavailable describe the same trigger."),
        p("Use the secondary region during a primary-region outage.", "positive_technical", "Equivalent failover behavior."),
        p("When the primary becomes unavailable, fail over to secondary.", "positive_word_order", "Same trigger and target."),
        n("Fail over to the secondary region when the primary is available.", "negative_operational", "Trigger condition reverses."),
        n("Fail over to the tertiary region when the primary is unavailable.", "negative_entity", "Failover destination changes."),
        n("Never fail over when the primary region is unavailable.", "negative_negation", "Failover action is negated."),
    )),
    Family("kafka_replication", "Use a replication factor of 3 for the Kafka orders topic.", (
        p("Set Kafka's orders topic replication factor to 3.", "positive_word_order", "Same platform, topic, and factor."),
        p("Replicate the Kafka orders topic across 3 brokers.", "positive_technical", "Equivalent replication objective in this fixture."),
        p("For the orders topic in Kafka, configure replication factor 3.", "positive_paraphrase", "Same configuration."),
        n("Use a replication factor of 2 for the Kafka orders topic.", "negative_quantity", "Replication factor changes."),
        n("Use a replication factor of 3 for the Kafka payments topic.", "negative_entity", "Topic changes."),
        n("Use a replication factor of 3 for the RabbitMQ orders queue.", "negative_entity", "Messaging product and resource type change."),
    )),
    Family("health_check", "Mark the service unhealthy after 3 failed health checks.", (
        p("After 3 health-check failures, mark the service unhealthy.", "positive_word_order", "Same threshold and outcome."),
        p("Declare the service unhealthy when 3 health checks fail.", "positive_synonym", "Declare and mark are equivalent."),
        p("Use 3 consecutive failed checks as the unhealthy threshold.", "positive_technical", "Same failure threshold in this fixture."),
        n("Mark the service unhealthy after 5 failed health checks.", "negative_quantity", "Failure threshold changes."),
        n("Mark the service healthy after 3 failed health checks.", "negative_operational", "Resulting health state reverses."),
        n("Never mark the service unhealthy after failed health checks.", "negative_negation", "Health transition is disabled."),
    )),
    Family("cors", "Allow CORS requests from https://app.example.com.", (
        p("Permit cross-origin requests from https://app.example.com.", "positive_synonym", "CORS expansion and permit retain the origin policy."),
        p("For https://app.example.com, allow CORS access.", "positive_word_order", "Same origin and permission."),
        p("CORS should accept the origin https://app.example.com.", "positive_technical", "Equivalent allowlist rule."),
        n("Allow CORS requests from https://admin.example.com.", "negative_entity", "Allowed origin changes."),
        n("Deny CORS requests from https://app.example.com.", "negative_permission", "Permission reverses."),
        n("Allow CORS requests from every origin.", "negative_permission", "Scope broadens from one origin to all origins."),
    )),
    Family("migration", "Apply database migration 42 before deploying release 8.1.", (
        p("Before release 8.1 is deployed, apply database migration 42.", "positive_word_order", "Same migration, release, and ordering."),
        p("Run migration 42 prior to deployment of version 8.1.", "positive_synonym", "Prior to and before are equivalent."),
        p("Database migration 42 must precede the 8.1 release deployment.", "positive_technical", "Same dependency."),
        n("Apply database migration 43 before deploying release 8.1.", "negative_quantity", "Migration identifier changes."),
        n("Apply database migration 42 before deploying release 8.2.", "negative_version", "Release version changes."),
        n("Apply database migration 42 after deploying release 8.1.", "negative_operational", "Operation order reverses."),
    )),
    Family("cache_invalidation", "Invalidate the product cache after a price update.", (
        p("After updating a price, clear the product cache.", "positive_synonym", "Clear and invalidate are equivalent here."),
        p("A price change should trigger product-cache invalidation.", "positive_technical", "Same trigger and action."),
        p("Clear cached products whenever their price is updated.", "positive_paraphrase", "Same invalidation policy."),
        n("Invalidate the product cache before a price update.", "negative_operational", "Ordering changes."),
        n("Invalidate the customer cache after a price update.", "negative_entity", "Cache target changes."),
        n("Keep the product cache after a price update.", "negative_operational", "Invalidation action reverses."),
    )),
    Family("docker_image", "Build the worker image from Python 3.12 slim.", (
        p("Use the Python 3.12 slim base for the worker image.", "positive_technical", "Same runtime version and image target."),
        p("The worker container should be based on slim Python 3.12.", "positive_word_order", "Same base image."),
        p("Create the worker image with the Python v3.12 slim base.", "positive_version_format", "Version notation changes only."),
        n("Build the worker image from Python 3.11 slim.", "negative_version", "Runtime version changes."),
        n("Build the API image from Python 3.12 slim.", "negative_entity", "Image target changes."),
        n("Build the worker image from Node.js 3.12 slim.", "negative_entity", "Runtime product changes."),
    )),
    Family("terraform_region", "Use Terraform to provision the network in AWS eu-west-1.", (
        p("Provision the AWS eu-west-1 network with Terraform.", "positive_word_order", "Same tool, provider, region, and resource."),
        p("With Terraform, create networking resources in AWS region eu-west-1.", "positive_technical", "Equivalent provisioning instruction."),
        p("Terraform should manage the network in AWS eu-west-1.", "positive_paraphrase", "Same infrastructure ownership."),
        n("Use Terraform to provision the network in AWS eu-central-1.", "negative_condition", "Region changes."),
        n("Use Terraform to provision the network in Azure eu-west-1.", "negative_entity", "Cloud provider changes."),
        n("Manually provision the network in AWS eu-west-1 without Terraform.", "negative_negation", "Provisioning mechanism changes."),
    )),
    Family("incident", "Is incident INC-1042 still open at severity 1?", (
        p("Does INC-1042 remain an open severity-1 incident?", "positive_word_order", "Same identifier, status, and severity."),
        p("Is severity 1 incident INC-1042 currently open?", "positive_paraphrase", "Same incident state."),
        p("For INC-1042, is the status open and severity set to 1?", "positive_technical", "Equivalent structured status question."),
        n("Is incident INC-1043 still open at severity 1?", "negative_identifier", "Incident identifier changes."),
        n("Is incident INC-1042 still open at severity 2?", "negative_quantity", "Severity changes."),
        n("Is incident INC-1042 closed at severity 1?", "negative_operational", "Status changes."),
    )),
    Family("http_method", "Does POST /v2/orders create a new order?", (
        p("Is a new order created by POST /v2/orders?", "positive_word_order", "Same method, endpoint, and behavior."),
        p("Will calling POST on /v2/orders create an order?", "positive_technical", "Equivalent API question."),
        p("Does the /v2/orders POST endpoint add a new order?", "positive_synonym", "Add and create are equivalent here."),
        n("Does GET /v2/orders create a new order?", "negative_permission", "HTTP method changes."),
        n("Does POST /v3/orders create a new order?", "negative_version", "Endpoint version changes."),
        n("Does POST /v2/customers create a new order?", "negative_entity", "Endpoint resource changes."),
    )),
    Family("tenant_balance", "What is my current account balance?", (
        Variant("Can you show my current account balance?", True, "positive_paraphrase", "Same user and account question.", "user", "alice"),
        Variant("How much money is currently in my account?", True, "positive_synonym", "Same user-specific intent.", "user", "alice"),
        Variant("My current account balance is what amount?", True, "positive_word_order", "Same user-specific intent.", "user", "alice"),
        Variant("What is my current account balance?", False, "negative_tenant", "Identical text belongs to a different user partition.", "user", "bob"),
        Variant("Can you show my current account balance?", False, "negative_tenant", "Paraphrase belongs to another user partition.", "user", "bob"),
        Variant("What is my current account balance?", False, "negative_tenant", "Identical text belongs to another tenant partition.", "user", "carol"),
    ), seed_scope="user", seed_user="alice"),
    Family("export_permission", "May tenant alpha export customer records?", (
        p("Is tenant alpha allowed to export customer records?", "positive_paraphrase", "Same tenant, action, and resource."),
        p("Can customer records be exported by tenant alpha?", "positive_word_order", "Same permission question."),
        p("Does tenant alpha have permission to export customer records?", "positive_technical", "Equivalent access question."),
        n("May tenant beta export customer records?", "negative_tenant", "Tenant identity changes."),
        n("May tenant alpha delete customer records?", "negative_permission", "Action changes."),
        n("Is tenant alpha forbidden from exporting customer records?", "negative_permission", "Permission polarity reverses."),
    )),
    Family("webhook", "Retry webhook delivery 4 times for API version 2.1.", (
        p("For API v2.1, make 4 webhook delivery attempts after failure.", "positive_technical", "Same API version and retry count."),
        p("Use 4 retries for webhook delivery in API 2.1.", "positive_word_order", "Same policy."),
        p("API version 2.1 should retry webhook delivery four times.", "positive_number_word", "Four and 4 express the same count."),
        n("Retry webhook delivery 5 times for API version 2.1.", "negative_quantity", "Retry count changes."),
        n("Retry webhook delivery 4 times for API version 2.2.", "negative_version", "API version changes."),
        n("Never retry webhook delivery for API version 2.1.", "negative_negation", "Retry behavior is negated."),
    )),
    Family("model_provider", "Route text embeddings to the OpenAI endpoint.", (
        p("Send text-embedding requests to the OpenAI endpoint.", "positive_technical", "Same workload and provider."),
        p("Use OpenAI as the endpoint for text embeddings.", "positive_word_order", "Same routing configuration."),
        p("The text embedding service should call OpenAI.", "positive_paraphrase", "Same provider choice."),
        n("Route text embeddings to the Azure endpoint.", "negative_entity", "Provider changes."),
        n("Route image generation to the OpenAI endpoint.", "negative_entity", "Workload changes."),
        n("Do not route text embeddings to the OpenAI endpoint.", "negative_negation", "Routing instruction is negated."),
    )),
    Family("prometheus_alert", "Alert when Prometheus reports CPU usage above 90 percent.", (
        p("When CPU usage exceeds 90 percent in Prometheus, send an alert.", "positive_word_order", "Same metric, threshold, and action."),
        p("Create a Prometheus alert for CPU usage greater than 90 percent.", "positive_technical", "Equivalent threshold rule."),
        p("Prometheus should alert on CPU utilization over 90 percent.", "positive_synonym", "Usage and utilization are equivalent here."),
        n("Alert when Prometheus reports CPU usage above 80 percent.", "negative_quantity", "Threshold changes."),
        n("Alert when Grafana reports CPU usage above 90 percent.", "negative_entity", "Monitoring product changes."),
        n("Do not alert when Prometheus reports CPU usage above 90 percent.", "negative_negation", "Alert action is negated."),
    )),
    Family("nginx_body", "Limit Nginx request bodies to 10 MB.", (
        p("Set the Nginx request-body limit to 10 MB.", "positive_technical", "Same product and size limit."),
        p("Nginx should accept request bodies up to 10 MB.", "positive_paraphrase", "Equivalent upper bound."),
        p("Use 10 MB as the maximum body size for Nginx requests.", "positive_word_order", "Same maximum."),
        n("Limit Nginx request bodies to 20 MB.", "negative_quantity", "Size limit changes."),
        n("Limit Nginx response bodies to 10 MB.", "negative_condition", "Direction changes from request to response."),
        n("Limit Cloudflare request bodies to 10 MB.", "negative_entity", "Product changes."),
    )),
)


FIELDS = [
    "query_id", "family_id", "split", "query", "seed_query", "seed_response",
    "expected_response", "reuse_eligible", "category", "provenance",
    "label_explanation", "seed_scope", "seed_user_id", "query_scope",
    "query_user_id", "contains_user_data",
]


def build_rows() -> list[dict[str, str]]:
    families = list(FAMILIES)
    shuffled_ids = [family.family_id for family in families]
    random.Random(RANDOM_SEED).shuffle(shuffled_ids)
    dev_ids = set(shuffled_ids[: len(families) // 4])
    rows: list[dict[str, str]] = []
    for family in families:
        split = "dev" if family.family_id in dev_ids else "test"
        variants = list(family.variants)
        if family.family_id in EXACT_REPEAT_FAMILIES:
            variants.append(Variant(
                family.seed_query, True, "positive_exact_repeat",
                "Exact repeated request with the same intended response.",
                family.seed_scope, family.seed_user,
            ))
        for index, variant in enumerate(variants, start=1):
            seed_response = f"response:{family.family_id}:seed"
            provenance = (
                "programmatic_exact_repeat_from_curated_seed"
                if variant.category == "positive_exact_repeat"
                else "manual_curated_v2_materialized_by_script"
            )
            rows.append({
                "query_id": f"{family.family_id}-{index:02d}",
                "family_id": family.family_id,
                "split": split,
                "query": variant.query,
                "seed_query": family.seed_query,
                "seed_response": seed_response,
                "expected_response": seed_response if variant.eligible else f"response:{family.family_id}:{index:02d}",
                "reuse_eligible": str(variant.eligible).lower(),
                "category": variant.category,
                "provenance": provenance,
                "label_explanation": variant.explanation,
                "seed_scope": family.seed_scope,
                "seed_user_id": family.seed_user,
                "query_scope": variant.query_scope,
                "query_user_id": variant.query_user,
                "contains_user_data": str(family.seed_scope == "user").lower(),
            })
    validate(rows)
    return rows


def validate(rows: list[dict[str, str]]) -> None:
    assert len(rows) >= 200, "V2 requires at least 200 credible cases"
    assert len({row["query_id"] for row in rows}) == len(rows), "query IDs must be unique"
    assert {row["reuse_eligible"] for row in rows} == {"true", "false"}
    family_splits: dict[str, set[str]] = {}
    for row in rows:
        family_splits.setdefault(row["family_id"], set()).add(row["split"])
        assert row["label_explanation"].strip()
        assert row["provenance"] in {
            "manual_curated_v2_materialized_by_script",
            "programmatic_exact_repeat_from_curated_seed",
        }
        if row["seed_scope"] == "user":
            assert row["seed_user_id"]
        if row["query_scope"] == "user":
            assert row["query_user_id"]
    assert all(len(splits) == 1 for splits in family_splits.values()), "family leakage detected"
    assert {row["split"] for row in rows} == {"dev", "test"}


def main() -> None:
    rows = build_rows()
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("dev", "test")}
    print(f"wrote {len(rows)} rows to {OUTPUT} ({counts})")


if __name__ == "__main__":
    main()
