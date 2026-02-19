param([string]$path)
$content = Get-Content $path
$content = $content -replace '^pick ([0-9a-f]+) Implemented Alerting Mechanism', 'reword $1 Initial commit'
$content | Set-Content $path
