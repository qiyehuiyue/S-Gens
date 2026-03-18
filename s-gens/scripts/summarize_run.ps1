param(
    [Parameter(Mandatory=$true)][string]$RunRoot,
    [string]$MetricsFile = ''
)

$cmd = @(
    'python', '-m', 'sgens.cli', 'summarize-run',
    '--run-root', $RunRoot
)

if ($MetricsFile -ne '') {
    $cmd += @('--metrics-file', $MetricsFile)
}

Write-Host "Summarizing run:" ($cmd -join ' ')
& $cmd[0] $cmd[1..($cmd.Length-1)]
