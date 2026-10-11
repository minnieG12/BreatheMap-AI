"""Verify the current AWS identity and write a small private S3 health-check object.

Example (Windows PowerShell):
  python scripts/verify_aws_s3.py --bucket YOUR_BUCKET --profile breathemap-ai --region ap-south-1

This writes one JSON object below <prefix>/healthchecks/. It does not make the
bucket or object public and does not read, print, or store credentials.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, ProfileNotFound


def main() -> int:
    parser = argparse.ArgumentParser(description="Check AWS credentials and S3 write access.")
    parser.add_argument("--bucket", required=True, help="Existing private S3 bucket name")
    parser.add_argument("--prefix", default="breathemap-ai", help="Object prefix (default: breathemap-ai)")
    parser.add_argument("--region", default=None, help="Bucket region, e.g. ap-south-1")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name; omit to use default credentials")
    args = parser.parse_args()

    try:
        session = boto3.Session(profile_name=args.profile or None, region_name=args.region or None)
        identity = session.client("sts").get_caller_identity()
        client = session.client("s3", region_name=args.region or session.region_name or None)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = args.prefix.strip().strip("/") or "breathemap-ai"
        key = f"{prefix}/healthchecks/connection-test-{timestamp}.json"
        payload = {
            "project": "BreatheMap AI",
            "check": "S3 write access",
            "status": "success",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "aws_account_id": identity.get("Account"),
            "aws_caller_arn": identity.get("Arn"),
            "bucket": args.bucket,
            "key": key,
            "note": "Private diagnostic object; no credentials are stored in this file.",
        }
        client.put_object(
            Bucket=args.bucket,
            Key=key,
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        print("AWS S3 WRITE CHECK PASSED")
        print(f"AWS account: ...{str(identity.get('Account', ''))[-4:]}")
        print(f"Bucket: {args.bucket}")
        print(f"Object: s3://{args.bucket}/{key}")
        print("Encryption: SSE-S3 (AES256)")
        print("Next: run Streamlit and upload the metrics + selected replay audit from the app.")
        return 0
    except NoCredentialsError:
        print("ERROR: AWS credentials not found. Configure an AWS CLI profile, SSO session, or IAM role.", file=sys.stderr)
    except ProfileNotFound:
        print("ERROR: AWS profile not found. Check the profile name or omit --profile to use the default chain.", file=sys.stderr)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        print(f"ERROR: AWS returned {error.get('Code', 'ClientError')}: {error.get('Message', 'request failed')}", file=sys.stderr)
        print("Check bucket name/region and s3:PutObject permission for the requested prefix.", file=sys.stderr)
    except BotoCoreError as exc:
        print(f"ERROR: AWS SDK error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
