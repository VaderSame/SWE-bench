# run.ps1
param (
    [string]$InstanceId = "astropy__astropy-12907",
    [string]$ContainerName = "swe_agnostic_env"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " [1/4] Resolving Instance: $InstanceId" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Fetch repo and base_commit dynamically from HuggingFace dataset
$MetaJson = python -c "from datasets import load_dataset; import json; ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test'); inst = next(x for x in ds if x['instance_id'] == '$InstanceId'); print(json.dumps({'repo': inst['repo'], 'base_commit': inst['base_commit'], 'image': inst['image']}))"
$Metadata =$MetaJson | ConvertFrom-Json

$Repo =$Metadata.repo
$BaseCommit =$Metadata.base_commit
$DOCKER_IMAGE=$Metadata.image
$Workspace = "$PWD\testbeds\$InstanceId"
$env:SWE_WORKSPACE=$Workspace

Write-Host "[+] Target Repository : https://github.com/$Repo.git"
Write-Host "[+] Target Base Commit: $BaseCommit"

# Clone if missing, reset to base commit
if (!(Test-Path $Workspace)) {
    Write-Host "[+] Cloning repository into $Workspace..."
    New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
    git clone "https://github.com/$Repo.git" $Workspace
}

Push-Location $Workspace
git clean -fdx
git reset --hard $BaseCommit
Pop-Location
Write-Host "[+] Repository reset to base_commit: $BaseCommit" -ForegroundColor Green

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " [2/4] Bootstrapping Agnostic Docker Environment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Cleanup existing container
$ExistingContainer = docker ps -a -q -f name="^${ContainerName}$"
if ($ExistingContainer) { docker rm -f $ContainerName | Out-Null }

$MountWorkspace = $Workspace.Replace('\', '/')
$MountBootstrap = ("$PWD\bootstrap_container.py").Replace('\', '/')

# TODO: check if $DOCKER_IMAGE is correct, otherwise create new docker image
# Start container with Debian bullseye (has gcc/make/git build toolchains)
Write-Host "[+] Starting base Python 3.9 container..."
docker run -d --name $ContainerName `
    -v "${MountWorkspace}:/workspace" `
    -v "${MountBootstrap}:/bootstrap_container.py" `
    -w /workspace `
    python:3.9-bullseye tail -f /dev/null | Out-Null

Write-Host "[+] Executing dynamic in-container dependency bootstrapper..." -ForegroundColor Yellow
docker exec $ContainerName python /bootstrap_container.py

$env:DOCKER_CONTAINER = $ContainerName
Write-Host "[+] Container environment ready and isolated." -ForegroundColor Green

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " [3/4] Launching LangGraph Agent" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

python runner.py --instance_id $InstanceId

$Stopwatch.Stop()
$ExitCode =$LASTEXITCODE

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " [4/4] Teardown & Results" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "[+] Removing container..."
$ExistingContainer = docker ps -a -q -f name="^${ContainerName}$"
if ($ExistingContainer) { docker rm -f $ContainerName | Out-Null }

if ($ExitCode -eq 0) {
    Write-Host " Script Execution : SUCCESS" -ForegroundColor Green
} else {
    Write-Host " Script Execution : FAILED (Exit Code: $ExitCode)" -ForegroundColor Red
}
Write-Host " Total Run Duration: $($Stopwatch.Elapsed.TotalSeconds.ToString('0.00')) seconds"
Write-Host "============================================================`n"