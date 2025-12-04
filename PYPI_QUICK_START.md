# PyPI Publishing Quick Start

Your package `airbornehrs` is ready for PyPI! Here's the fastest path:

## Quick Commands

### 1️⃣ Create PyPI Account
- Go to https://pypi.org/account/register/
- Verify email
- Create API token: https://pypi.org/manage/account/tokens/

### 2️⃣ Store Credentials

**Option A: PowerShell Environment Variables** (easiest)
```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-AgEIcHlwaS5vcmc..."  # Your token
```

**Option B: pip.ini**
Create `%APPDATA%\pip\pip.ini`:
```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

### 3️⃣ Test Upload to Test PyPI (Recommended)
```powershell
cd "c:\Users\surya\In Use\Personal\UltOrg\Airborne.HRS\MirrorMind"
python -m twine upload --repository testpypi dist/*
```

Then test installation:
```powershell
pip install -i https://test.pypi.org/simple/ airbornehrs
python -c "from airbornehrs import AdaptiveFramework; print('✅ Works!')"
```

### 4️⃣ Upload to Production PyPI
```powershell
python -m twine upload dist/*
```

### 5️⃣ Verify Live Installation
```powershell
python -m pip uninstall airbornehrs
pip install airbornehrs
python -c "from airbornehrs import AdaptiveFramework; print('✅ Success')"
```

## Done! 🎉

Your package is now live at: **https://pypi.org/project/airbornehrs/**

Users can now install with:
```bash
pip install airbornehrs
```

---

## Files Included

- `PYPI_PUBLISH.md` - Detailed publishing guide
- `pypi_upload.py` - Simple upload script
- `pypi_interactive_guide.py` - Interactive setup wizard

---

## Next Release

To publish a new version:

1. Edit `pyproject.toml` and increment version:
   ```toml
   version = "0.1.1"
   ```

2. Rebuild:
   ```powershell
   rm -r build dist
   python -m build
   ```

3. Upload:
   ```powershell
   python -m twine upload dist/*
   ```

That's it! 🚀
