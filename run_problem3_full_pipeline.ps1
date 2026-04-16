$ErrorActionPreference = "Stop"
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
  $PSNativeCommandUseErrorActionPreference = $false
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
$syncPython = Join-Path $projectRoot ".venv-sync314\Scripts\python.exe"
$script = Join-Path $projectRoot "problem3_cnn_qnn.py"
$uploader = Join-Path $projectRoot "upload_problem3_run_to_wandb.py"

$seed = 12505009
$epochs = 10
$numWorkers = 0
$wandbMode = "disabled"
$wandbEntity = "pegasus2737-national-taiwan-university"
$wandbProject = "HW1-3"

$baselineBatchSize = 64
$qnnBatchSize = 32

$fullTrain = 50000
$fullTest = 10000

$baselineOut = "baseline_full_frozen_e10"
$qnnOut = "qnn_full_batched_e10"

$baselineCheckpoint = Join-Path $projectRoot "outputs\problem3\$baselineOut\mlp\backbone_state.pt"
$baselineRunDir = Join-Path $projectRoot "outputs\problem3\$baselineOut"
$qnnRunDir = Join-Path $projectRoot "outputs\problem3\$qnnOut"

function Invoke-NativeProcess {
  param(
    [string]$Exe,
    [string[]]$Arguments
  )

  $stdoutPath = [System.IO.Path]::GetTempFileName()
  $stderrPath = [System.IO.Path]::GetTempFileName()
  try {
    $quotedArgs = $Arguments | ForEach-Object {
      if ($_ -match '[\s"]') {
        '"' + ($_ -replace '"', '\"') + '"'
      }
      else {
        $_
      }
    }
    $proc = Start-Process `
      -FilePath $Exe `
      -ArgumentList ($quotedArgs -join ' ') `
      -NoNewWindow `
      -Wait `
      -PassThru `
      -RedirectStandardOutput $stdoutPath `
      -RedirectStandardError $stderrPath

    if (Test-Path $stdoutPath) {
      Get-Content $stdoutPath | ForEach-Object { Write-Host $_ }
    }
    if (Test-Path $stderrPath) {
      Get-Content $stderrPath | ForEach-Object { Write-Host $_ }
    }

    return [int]$proc.ExitCode
  }
  finally {
    Remove-Item $stdoutPath, $stderrPath -ErrorAction SilentlyContinue
  }
}

function Upload-ToWandb {
  param(
    [string]$RunDir,
    [string]$RunName
  )

  Write-Host "Uploading $RunName to wandb via Python 3.14 sync env..."
  $exitCode = Invoke-NativeProcess -Exe $syncPython -Arguments @(
    $uploader,
    "--run-dir", $RunDir,
    "--entity", $wandbEntity,
    "--project", $wandbProject,
    "--run-name", $RunName
  )
  if ($exitCode -ne 0) {
    throw "wandb upload failed for $RunName"
  }
}

Write-Host "Running Problem 3 full baseline..."
$exitCode = Invoke-NativeProcess -Exe $python -Arguments @(
  $script,
  "--seed", $seed,
  "--epochs", $epochs,
  "--batch-size", $baselineBatchSize,
  "--subset-train", $fullTrain,
  "--subset-test", $fullTest,
  "--num-workers", $numWorkers,
  "--run-baseline",
  "--wandb", $wandbMode,
  "--wandb-entity", $wandbEntity,
  "--wandb-project", $wandbProject,
  "--output-dir", $baselineOut
)
if ($exitCode -ne 0) {
  throw "Baseline run failed with exit code $exitCode"
}

if (-not (Test-Path $baselineCheckpoint)) {
  throw "Baseline checkpoint not found: $baselineCheckpoint"
}

if (-not (Test-Path (Join-Path $baselineRunDir "summary.json"))) {
  throw "Baseline summary not found: $(Join-Path $baselineRunDir 'summary.json')"
}

Upload-ToWandb -RunDir $baselineRunDir -RunName $baselineOut

Write-Host "Running Problem 3 full QNN..."
$exitCode = Invoke-NativeProcess -Exe $python -Arguments @(
  $script,
  "--seed", $seed,
  "--epochs", $epochs,
  "--batch-size", $qnnBatchSize,
  "--subset-train", $fullTrain,
  "--subset-test", $fullTest,
  "--num-workers", $numWorkers,
  "--run-qnn",
  "--freeze-backbone",
  "--backbone-checkpoint", $baselineCheckpoint,
  "--wandb", $wandbMode,
  "--wandb-entity", $wandbEntity,
  "--wandb-project", $wandbProject,
  "--qnn-head-type", "residual",
  "--qnn-device", "lightning.qubit",
  "--qnn-units", 12,
  "--qnn-layers", 2,
  "--qnn-hidden-dim", 64,
  "--qnn-bottleneck-dim", 64,
  "--output-dir", $qnnOut
)
if ($exitCode -ne 0) {
  throw "QNN run failed with exit code $exitCode"
}

if (-not (Test-Path (Join-Path $qnnRunDir "summary.json"))) {
  throw "QNN summary not found: $(Join-Path $qnnRunDir 'summary.json')"
}

Upload-ToWandb -RunDir $qnnRunDir -RunName $qnnOut

Write-Host "Problem 3 full pipeline complete."
