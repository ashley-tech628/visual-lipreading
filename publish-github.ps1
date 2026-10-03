param(
    [string]$Repository = 'ashley-tech628/visual-lipreading',
    [ValidateSet('public','private')][string]$Visibility = 'public'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
Set-Location -LiteralPath $projectRoot
Get-Command git, gh -ErrorAction Stop | Out-Null
& git config --global --add safe.directory $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not trust this specific local project directory.' }
& gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Sign in with gh auth login, then rerun this script.' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.git'))) {
    & git -c "safe.directory=$projectRoot" init -b main
    if ($LASTEXITCODE -ne 0) { throw 'Git initialization failed.' }
}
& git -c "safe.directory=$projectRoot" add .
if ($LASTEXITCODE -ne 0) { throw 'Git staging failed.' }
& git -c "safe.directory=$projectRoot" diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    & git -c "safe.directory=$projectRoot" commit -m 'Add visual lipreading research, fusion models and reproducible pipeline'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed. Check your Git user.name and user.email.' }
}
# A missing repository is an expected probe result. Windows PowerShell can
# promote native stderr to a terminating error even when it is redirected.
$savedErrorPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & gh repo view $Repository --json nameWithOwner 2>$null
    $repositoryExists = ($LASTEXITCODE -eq 0)
} finally {
    $ErrorActionPreference = $savedErrorPreference
}
if (-not $repositoryExists) {
    & gh repo create $Repository "--$Visibility" --source $projectRoot --remote origin
    if ($LASTEXITCODE -ne 0) { throw 'Repository creation failed.' }
} else {
    try {
        $ErrorActionPreference = 'Continue'
        $origin = & git -c "safe.directory=$projectRoot" remote get-url origin 2>$null
        $originExists = ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $savedErrorPreference
    }
    if (-not $originExists) {
        & git -c "safe.directory=$projectRoot" remote add origin "https://github.com/$Repository.git"
        if ($LASTEXITCODE -ne 0) { throw 'Adding the origin remote failed.' }
    } elseif ($origin -notmatch [regex]::Escape($Repository)) {
        throw "Existing origin is different: $origin. Review it before pushing."
    }
}
& git -c "safe.directory=$projectRoot" push -u origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed. No force push was attempted.' }
Write-Host "Published: https://github.com/$Repository"
Write-Host 'Local checkpoint binaries remain excluded from Git; publish them as a reviewed release asset if desired.'
