# run.ps1
param (
    [string]$InstanceId = "astropy__astropy-12907",
    [string]$ContainerName = "swe_agnostic_env",
    [switch]$KeepContainer   # leave the container running afterwards, e.g. to `docker exec -it $ContainerName bash`
)

$ErrorActionPreference = "Stop"

# $ErrorActionPreference does not cover native commands, so check exit codes by hand.
function Assert-Native([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Remove-SweContainer {
    $existing = docker ps -a -q -f "name=^${ContainerName}$"
    if ($existing) { docker rm -f $ContainerName | Out-Null }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " [1/4] Resolving Instance: $InstanceId" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# This copy of SWE-bench Lite has the `image` column; princeton-nlp/SWE-bench_Lite does not.
$MetaJson = python -c "from datasets import load_dataset; import json; ds = load_dataset('SWE-bench/SWE-bench_Lite', split='test'); inst = next(x for x in ds if x['instance_id'] == '$InstanceId'); print(json.dumps({'repo': inst['repo'], 'base_commit': inst['base_commit'], 'image': inst['image']}))" | Select-Object -Last 1
Assert-Native "Dataset lookup for '$InstanceId'"
$Metadata = $MetaJson | ConvertFrom-Json

$DockerImage = $Metadata.image
if (-not $DockerImage) { throw "Dataset row for '$InstanceId' has no image." }

Write-Host "[+] Repository  : $($Metadata.repo)"
Write-Host "[+] Base commit : $($Metadata.base_commit)"
Write-Host "[+] Image       : $DockerImage"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " [2/4] Starting the official SWE-bench container" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Remove-SweContainer

# The images are x86_64 only; --platform makes Docker emulate on ARM hosts instead of failing.
Write-Host "[+] Pulling image (large on the first run)..."
docker pull --platform linux/amd64 $DockerImage
Assert-Native "docker pull $DockerImage (check your network / Docker login, or build the image locally with the SWE-bench harness)"

# No bind mounts: the repo lives at /testbed inside the image and is only touched through `docker exec`.
docker run -d --name $ContainerName --platform linux/amd64 $DockerImage tail -f /dev/null | Out-Null
Assert-Native "docker run"

$env:DOCKER_CONTAINER = $ContainerName
Write-Host "[+] Container '$ContainerName' is running." -ForegroundColor Green

$ExitCode = 1
try {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host " [3/4] Launching LangGraph Agent" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    python runner.py --instance_id $InstanceId
    $ExitCode = $LASTEXITCODE
    $Stopwatch.Stop()
}
finally {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host " [4/4] Teardown & Results" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    if ($KeepContainer) {
        Write-Host "[+] Keeping container '$ContainerName' (remove with: docker rm -f $ContainerName)"
    } else {
        Write-Host "[+] Removing container..."
        Remove-SweContainer
    }
}

if ($ExitCode -eq 0) {
    Write-Host " Script Execution : SUCCESS" -ForegroundColor Green
} else {
    Write-Host " Script Execution : FAILED (Exit Code: $ExitCode)" -ForegroundColor Red
}
Write-Host " Total Run Duration: $($Stopwatch.Elapsed.TotalSeconds.ToString('0.00')) seconds"
Write-Host "============================================================`n"