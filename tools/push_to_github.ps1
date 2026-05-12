<#
.SYNOPSIS
    Push the local repo to GitHub. Run from project root.

.DESCRIPTION
    Two modes, depending on whether `gh` CLI is available:

    Mode A — gh CLI (fully automated):
        gh repo create <name> --public --source=. --remote=origin --push

    Mode B — manual (no gh):
        1. Create the repo manually on github.com (do NOT initialise it with README/LICENSE).
        2. Run:
             git remote add origin https://github.com/<username>/<repo>.git
             git branch -M main
             git push -u origin main

.PARAMETER RepoName
    Name of the GitHub repository to create.

.PARAMETER Visibility
    'public' or 'private'.

.PARAMETER Username
    Your GitHub username (only required for Mode B URL construction).

.EXAMPLE
    .\tools\push_to_github.ps1 -RepoName bank-distributed-ml -Visibility public -Username yourname
#>

param(
    [Parameter(Mandatory=$true)][string]$RepoName,
    [ValidateSet('public','private')][string]$Visibility = 'public',
    [string]$Username = ''
)

$ErrorActionPreference = 'Stop'

Write-Host "Pushing repo '$RepoName' as $Visibility ..." -ForegroundColor Cyan

$ghAvailable = $null -ne (Get-Command gh -ErrorAction SilentlyContinue)

if ($ghAvailable) {
    Write-Host "gh CLI detected — using fully automated path." -ForegroundColor Green
    gh repo create $RepoName --$Visibility --source=. --remote=origin --push
    if ($LASTEXITCODE -ne 0) {
        Write-Error "gh repo create failed."
        exit 1
    }
    Write-Host "Done. Repo URL:" -ForegroundColor Green
    gh repo view --web
} else {
    if (-not $Username) {
        Write-Error "gh CLI not installed and -Username not provided. Either:`n  (a) Install gh:  winget install --id GitHub.cli`n  (b) Re-run with -Username <your-github-username>"
        exit 1
    }
    $url = "https://github.com/$Username/$RepoName.git"
    Write-Host "gh CLI not found. Falling back to manual push to: $url" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "BEFORE running this script you must:" -ForegroundColor Yellow
    Write-Host "  1. Go to https://github.com/new" -ForegroundColor Yellow
    Write-Host "  2. Create '$RepoName' as $Visibility" -ForegroundColor Yellow
    Write-Host "  3. Do NOT initialise it with README, .gitignore, or LICENSE" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter when the empty repo is created"

    git remote add origin $url
    git branch -M main
    git push -u origin main
    Write-Host "Pushed to $url" -ForegroundColor Green
}
