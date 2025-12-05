#!/usr/bin/env powershell
<#
.SYNOPSIS
    Simple PyPI credential setup for Windows
    
.DESCRIPTION
    Sets environment variables for PyPI upload
    
.PARAMETER Token
    Your PyPI API token (from https://pypi.org/manage/account/tokens/)
    
.EXAMPLE
    .\setup_pypi.ps1 -Token "pypi-AgE..."
#>

param(
    [string]$Token
)

if (-not $Token) {
    Write-Host "`n❌ PyPI token required!`n"
    Write-Host "Get your token from: https://pypi.org/manage/account/tokens/`n"
    Write-Host "Usage:`n  .\setup_pypi.ps1 -Token 'pypi-AgE...'`n"
    exit 1
}

Write-Host "`n📝 Setting up PyPI credentials...`n"

# Set environment variables for current session
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = $Token

Write-Host "✅ Environment variables set for this session:`n"
Write-Host "   TWINE_USERNAME = __token__"
Write-Host "   TWINE_PASSWORD = $($Token.Substring(0, 20))..."

# Create pip.ini for persistent storage
$pipConfigPath = "$env:APPDATA\pip\pip.ini"

if (-not (Test-Path "$env:APPDATA\pip")) {
    New-Item -ItemType Directory -Path "$env:APPDATA\pip" -Force | Out-Null
}

$pipConfig = @"
[distutils]
index-servers =
    pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = $Token
"@

# Backup existing config
if (Test-Path $pipConfigPath) {
    Copy-Item $pipConfigPath "$pipConfigPath.bak"
    Write-Host "`n   (Backed up existing pip.ini to pip.ini.bak)`n"
}

# Write new config
$pipConfig | Out-File -FilePath $pipConfigPath -Encoding UTF8
Write-Host "✅ Credentials saved to: $pipConfigPath`n"

# Test twine
Write-Host "🧪 Testing twine command...`n"
$result = & python -m twine --version
if ($?) {
    Write-Host "✅ $result`n"
} else {
    Write-Host "❌ Twine not found. Installing...`n"
    python -m pip install twine --quiet
    Write-Host "✅ Twine installed`n"
}

Write-Host "`n📚 Next steps:`n"
Write-Host "  1. Test upload: python -m twine upload --repository testpypi dist/*"
Write-Host "  2. Production: python -m twine upload dist/*`n"

Write-Host "📖 For help, see: PYPI_QUICK_START.md`n"
