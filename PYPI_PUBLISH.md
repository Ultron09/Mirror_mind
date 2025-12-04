# Publishing airbornehrs to PyPI

This guide walks through publishing your package to PyPI (Python Package Index) and testing the installation.

## Step 1: Create PyPI Account

If you don't have a PyPI account:

1. Go to https://pypi.org/account/register/
2. Create an account with your email
3. Verify your email
4. Enable 2-factor authentication (recommended)
5. Create an API token at https://pypi.org/manage/account/tokens/
   - Click "Add API token"
   - Name: `airbornehrs-upload` (or your choice)
   - Scope: "Entire account"
   - Copy the token (you'll only see it once!)

## Step 2: Store PyPI Credentials

On Windows, create a file: `%APPDATA%\pip\pip.ini`

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your actual PyPI token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgE...  # Your test.pypi token (optional)
```

OR use environment variables:
```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-AgEIcHlwaS5vcmc..."
```

## Step 3: Test Upload to Test PyPI (Optional but Recommended)

First test on https://test.pypi.org to verify everything works:

1. Create a Test PyPI account at https://test.pypi.org/account/register/
2. Create a test API token
3. Upload:

```powershell
cd c:\Users\surya\In Use\Personal\UltOrg\Airborne.HRS\MirrorMind
python -m twine upload --repository testpypi dist/*
```

Expected output:
```
Uploading airbornehrs-0.1.0.tar.gz
Uploading airbornehrs-0.1.0-py3-none-any.whl
View at:
https://test.pypi.org/project/airbornehrs/
```

4. Test installation:
```powershell
python -m pip install -i https://test.pypi.org/simple/ airbornehrs
```

## Step 4: Validate Before Production Upload

Check your package quality:

```powershell
python -m twine check dist/*
```

Expected: `✓ airbornehrs-0.1.0-py3-none-any.whl: PASSED`

## Step 5: Upload to Production PyPI

Once everything is tested:

```powershell
cd c:\Users\surya\In Use\Personal\UltOrg\Airborne.HRS\MirrorMind
python -m twine upload dist/*
```

Enter your PyPI username and password when prompted (or use token credentials).

Expected output:
```
Uploading airbornehrs-0.1.0.tar.gz
Uploading airbornehrs-0.1.0-py3-none-any.whl
View at:
https://pypi.org/project/airbornehrs/
```

## Step 6: Test Production Installation

In a fresh environment:

```powershell
# Create test venv
python -m venv test_install_venv
.\test_install_venv\Scripts\Activate.ps1

# Install from PyPI
pip install airbornehrs

# Test import
python -c "from airbornehrs import AdaptiveFramework; print('✅ Success')"
```

## Step 7: Update Version for Next Release

Edit `pyproject.toml`:

```toml
version = "0.1.1"  # Increment version
```

Rebuild:
```powershell
rm -r build dist
python -m build
python -m twine upload dist/*
```

---

## Quick Reference Commands

**Check package validity:**
```powershell
python -m twine check dist/*
```

**Upload to Test PyPI:**
```powershell
python -m twine upload --repository testpypi dist/*
```

**Upload to Production PyPI:**
```powershell
python -m twine upload dist/*
```

**Install from PyPI:**
```powershell
pip install airbornehrs
```

**Install specific version:**
```powershell
pip install airbornehrs==0.1.0
```

**Upgrade:**
```powershell
pip install --upgrade airbornehrs
```

**Test install before upload:**
```powershell
pip install dist/airbornehrs-0.1.0-py3-none-any.whl
```

---

## Troubleshooting

**"Invalid token"**
- Check your API token is correct
- Make sure you used `__token__` as username
- Token format should be `pypi-AgE...`

**"File already exists"**
- You can't upload the same version twice
- Increment version in `pyproject.toml`
- Rebuild with `python -m build`

**"Invalid credentials"**
- Check `pip.ini` or environment variables
- Make sure no extra spaces in token

**Package not appearing immediately**
- PyPI caches may take 5-15 minutes
- Check https://pypi.org/project/airbornehrs/

---

## Next: GitHub Actions for Auto-Deploy

Once working, set up GitHub Actions to auto-upload on release:

1. Add PyPI token to GitHub Secrets
2. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI
on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.11
      - run: pip install build twine
      - run: python -m build
      - run: python -m twine upload dist/* -u __token__ -p ${{ secrets.PYPI_API_TOKEN }}
```

Then releases auto-publish!
