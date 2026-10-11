# Configure and verify the AWS S3 integration

I use Amazon S3 to store a private, timestamped audit trail for a forecast replay. The app uploads three JSON files per run: evaluation metrics, a selected forecast audit, and a manifest. It does not make the bucket public.

> AWS Builder Center student verification is not, by itself, an AWS cloud account or S3 credential. This setup requires access to an AWS account with permission to write to the bucket.

## 1. Create a private S3 bucket

1. Open the [Amazon S3 console](https://console.aws.amazon.com/s3/).
2. Create a bucket with a globally unique name. Choose the region you intend to use (for example, `ap-south-1`, Mumbai).
3. Keep **Block all public access** enabled.
4. Keep default encryption enabled. The app also sends `ServerSideEncryption="AES256"` with each upload.
5. Do not add a public bucket policy or public object ACLs.

See AWS guidance: [create a bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) and [S3 access control](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/access-control-overview.html).

## 2. Grant only the required write permission

For the app's default prefix, the principal needs `s3:PutObject` for `arn:aws:s3:::YOUR_BUCKET_NAME/breathemap-ai/*`. Replace `YOUR_BUCKET_NAME` with the real bucket name before saving the policy. If you change the app's prefix, update the resource ARN to match.

Example minimal object-write policy statement:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BreatheMapWriteAuditObjects",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/breathemap-ai/*"
    }
  ]
}
```

The app does not need public-read permissions. Follow your account's policy for using a role, SSO, or temporary credentials. If a local profile is the practical option for your demo, keep it in your local AWS configuration directory—not in the repository.

## 3. Configure a local AWS profile

If AWS CLI is installed, check it:

```powershell
aws --version
```

Configure a profile using credentials issued for your account:

```powershell
aws configure --profile breathemap-ai
```

The CLI prompts for access key, secret key, default region, and output format. Use a least-privilege identity. Never paste keys into this app or ChatGPT; never commit them to GitHub. The CLI stores profile data outside this project, normally under your user home directory.

Verify that the selected identity is available:

```powershell
aws sts get-caller-identity --profile breathemap-ai
```

In the same PowerShell window where you will run Streamlit, set the profile and region (replace the region if your bucket uses a different one):

```powershell
$env:AWS_PROFILE = "breathemap-ai"
$env:AWS_DEFAULT_REGION = "ap-south-1"
```

You can also copy `.env.example` to `.env` to prefill the bucket name, prefix, and region in the app. The `.env` file must remain local and ignored by Git. **Do not put access keys or secret keys into `.env`.**

## 4. Test a real S3 write before opening the app

From the repository root, with `.venv` activated:

```powershell
python scripts/verify_aws_s3.py --bucket YOUR_BUCKET_NAME --profile breathemap-ai --region ap-south-1
```

Replace the bucket and region with your own. On success, the script prints `AWS S3 WRITE CHECK PASSED` and the URI of a small diagnostic JSON object written under `breathemap-ai/healthchecks/`. Open the bucket in the AWS console and confirm the object appears.

If the script fails:

- **No credentials / profile not found:** check the profile name and run `aws sts get-caller-identity --profile breathemap-ai` again.
- **AccessDenied:** check `s3:PutObject` on the selected bucket/prefix.
- **Wrong region / redirect:** pass the bucket's exact region in `--region` and the app's AWS region field.
- **No bucket access:** confirm the bucket name and account/region.

## 5. Demonstrate the app integration

Start the app from the same terminal, so it inherits `AWS_PROFILE` and `AWS_DEFAULT_REGION`:

```powershell
streamlit run app.py
```

Scroll to **AWS · S3 audit trail**, enter the bucket, prefix, and region, and click **Upload evaluation + replay audit to S3**. The app should confirm success and show three S3 URIs for the timestamped run.

In the S3 console, open `breathemap-ai/runs/<timestamp>/` and verify:

- `model_metrics.json`
- `forecast_audit.json`
- `manifest.json`

Show both the app success message and the objects in the bucket in the hackathon demo. Do not show credential files or secrets on screen.

## 6. Cost and cleanup

This feature uploads small JSON files, not the full dataset or model. Keep the bucket private and review AWS billing/Free Tier status. When you no longer need the demo data, remove test objects and delete the bucket if it was created only for this project and is empty. Never delete a shared or pre-existing bucket as part of cleanup.
