param(
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$Raw,
    [Parameter(Mandatory=$true)][string]$Kg,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Preset = 'paper',
    [int]$MaxExamples = 0,
    [string]$ExperimentName = ''
)

$cmd = @(
    'python', '-m', 'sgens.cli', 'run-dataset',
    '--dataset', $Dataset,
    '--raw', $Raw,
    '--kg', $Kg,
    '--output-dir', $OutputDir,
    '--preset', $Preset
)

if ($MaxExamples -gt 0) {
    $cmd += @('--max-examples', "$MaxExamples")
}

if ($ExperimentName -ne '') {
    $cmd += @('--experiment-name', $ExperimentName)
}

Write-Host "Running S-Gens pipeline:" ($cmd -join ' ')
& $cmd[0] $cmd[1..($cmd.Length-1)]
