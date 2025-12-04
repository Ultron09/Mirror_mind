#!/usr/bin/env python
"""
Interactive PyPI setup and upload guide
Walks you through the entire process step-by-step
"""

import sys
import os
import subprocess
from pathlib import Path


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(num, title):
    """Print formatted step"""
    print(f"\n📌 STEP {num}: {title}")
    print("-" * 70)


def ask_yes_no(question):
    """Ask yes/no question"""
    while True:
        response = input(f"\n❓ {question} (yes/no): ").lower().strip()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        print("   Please enter 'yes' or 'no'")


def check_pypi_account():
    """Guide user to create PyPI account"""
    print_step(1, "Create PyPI Account")
    
    if ask_yes_no("Do you already have a PyPI account?"):
        print("   ✅ Great! Skip to next step")
        return True
    
    print("""
   📝 To create an account:
   
   1. Go to: https://pypi.org/account/register/
   2. Fill in your details and create account
   3. Verify your email
   4. Enable 2FA (recommended)
   5. Create API token at: https://pypi.org/manage/account/tokens/
      - Click "Add API token"
      - Name: airbornehrs-upload
      - Scope: Entire account
      - COPY THE TOKEN (only shown once!)
   
   ⏸️  Once done, press Enter to continue...
    """)
    input()
    return True


def setup_credentials():
    """Help user set up PyPI credentials"""
    print_step(2, "Store PyPI Credentials")
    
    print("""
   Choose how to store credentials:
   
   Option A: pip.ini file (Windows)
   - Location: %APPDATA%\\pip\\pip.ini
   - Format:
     [distutils]
     index-servers = pypi
     [pypi]
     repository = https://upload.pypi.org/legacy/
     username = __token__
     password = pypi-AgEIcHlwaS5vcmc...
   
   Option B: Environment variables
   - $env:TWINE_USERNAME = "__token__"
   - $env:TWINE_PASSWORD = "pypi-AgE..."
   
   Option C: Use keyring (recommended)
   - Run: python -m pip install keyring
   - Then: keyring set https://upload.pypi.org/legacy/ __token__
     - Enter your token when prompted
    """)
    
    choice = input("\n💾 Which option? (A/B/C, default=A): ").strip().upper()
    
    if choice == "C":
        print("\n   Installing keyring...")
        subprocess.run([sys.executable, "-m", "pip", "install", "keyring", "--quiet"])
        print("   ✅ Keyring installed")
        return True
    elif choice == "B":
        print("""
   📌 Set environment variables in PowerShell:
   
   $env:TWINE_USERNAME = "__token__"
   $env:TWINE_PASSWORD = "pypi-AgE..."
   
   ✅ Press Enter when done...
        """)
        input()
        return True
    else:  # A or default
        print("""
   📝 Edit or create: %APPDATA%\\pip\\pip.ini
   
   Add content:
   [distutils]
   index-servers = pypi
   [pypi]
   repository = https://upload.pypi.org/legacy/
   username = __token__
   password = pypi-YOUR-TOKEN-HERE
   
   ✅ Press Enter when done...
        """)
        input()
        return True


def test_package():
    """Test uploading to Test PyPI"""
    print_step(3, "Test Upload to Test PyPI (Recommended)")
    
    print("""
   Test PyPI allows you to verify everything works before
   uploading to production PyPI.
   
   Steps:
   1. Create Test PyPI account: https://test.pypi.org/account/register/
   2. Create API token at: https://test.pypi.org/manage/account/tokens/
   3. Add to pip.ini or environment:
      [testpypi]
      repository = https://test.pypi.org/legacy/
      username = __token__
      password = pypi-...
    """)
    
    if ask_yes_no("Do you want to test-upload now?"):
        print("\n   Uploading to Test PyPI...")
        project_dir = Path(__file__).parent
        result = subprocess.run([
            sys.executable, "-m", "twine", "upload",
            "--repository", "testpypi",
            str(project_dir / "dist" / "*")
        ], cwd=project_dir)
        
        if result.returncode == 0:
            print("\n   ✅ Test upload successful!")
            print("\n   To test installation:")
            print("   pip install -i https://test.pypi.org/simple/ airbornehrs")
            input("\n   Press Enter after testing (or skip)...")
        else:
            print("\n   ❌ Test upload failed")
            return False
    
    return True


def production_upload():
    """Upload to production PyPI"""
    print_step(4, "Production Upload")
    
    print("""
   ⚠️  YOU ARE ABOUT TO UPLOAD TO PRODUCTION PyPI
   
   This means:
   - The package will be publicly available
   - Anyone can install it: pip install airbornehrs
   - Version cannot be modified or deleted (by design)
   - To fix issues, you must release a new version
   
   Make sure everything is ready!
    """)
    
    if ask_yes_no("Continue with production upload?"):
        print("\n   🚀 Uploading to Production PyPI...")
        project_dir = Path(__file__).parent
        result = subprocess.run([
            sys.executable, "-m", "twine", "upload",
            str(project_dir / "dist" / "*")
        ], cwd=project_dir)
        
        if result.returncode == 0:
            print("\n   ✅ Production upload successful!")
            print("\n   🎉 Package is now live at: https://pypi.org/project/airbornehrs/")
            print("\n   To install:")
            print("   pip install airbornehrs")
            return True
        else:
            print("\n   ❌ Production upload failed")
            return False
    
    print("   ⏭️  Skipping production upload for now")
    return True


def verify_installation():
    """Help user verify installation"""
    print_step(5, "Verify Installation")
    
    print("""
   Create a fresh virtual environment and test:
   
   PowerShell:
   python -m venv test_env
   .\\test_env\\Scripts\\Activate.ps1
   pip install airbornehrs
   python -c "from airbornehrs import AdaptiveFramework; print('✅ Success')"
   deactivate
    """)
    
    if ask_yes_no("Run verification test now?"):
        print("\n   Testing in fresh environment...")
        # Create venv and test
        test_venv = Path(__file__).parent / "test_final_env"
        
        # Create venv
        subprocess.run([sys.executable, "-m", "venv", str(test_venv)])
        
        # Activate and install
        python_exe = test_venv / "Scripts" / "python.exe"
        subprocess.run([str(python_exe), "-m", "pip", "install", "airbornehrs", "--quiet"])
        
        # Test import
        result = subprocess.run([
            str(python_exe), "-c",
            "from airbornehrs import AdaptiveFramework; print('✅ Installation verified!')"
        ])
        
        if result.returncode == 0:
            print("\n   ✅ Verification successful!")
        else:
            print("\n   ⚠️  Verification had issues")
    
    return True


def main():
    """Main flow"""
    print_header("🚀 PyPI Upload Guide for airbornehrs")
    
    print("""
   This interactive guide will help you:
   1. Create a PyPI account
   2. Set up credentials
   3. Test upload to Test PyPI
   4. Upload to production PyPI
   5. Verify installation
   
   Let's begin!
    """)
    
    # Step 1: Account
    if not check_pypi_account():
        print("\n❌ Please create an account first")
        return
    
    # Step 2: Credentials
    if not setup_credentials():
        print("\n❌ Failed to setup credentials")
        return
    
    # Step 3: Test upload
    if not test_package():
        print("\n⚠️  Test upload skipped or failed")
        if not ask_yes_no("Continue to production?"):
            print("\n✅ Exiting. Run this script again when ready.")
            return
    
    # Step 4: Production upload
    if not production_upload():
        print("\n❌ Production upload failed")
        return
    
    # Step 5: Verify
    verify_installation()
    
    print_header("✅ All Done!")
    print("""
   Your package is ready!
   
   📦 PyPI: https://pypi.org/project/airbornehrs/
   📖 API: https://github.com/Ultron09/Mirror_mind#api-reference
   🔗 GitHub: https://github.com/Ultron09/Mirror_mind
   
   Next steps:
   - Share the package: pip install airbornehrs
   - Update version for next release
   - Consider GitHub Actions for auto-deploy
   
   Thank you for using airbornehrs! 🎉
    """)


if __name__ == "__main__":
    main()
