$projectPath = "E:\Users\zc\Desktop\FEVR\mlp_hyperparameter_search"

$lastSearchDate = ""
$lastTradeTime = ""

Write-Host "FEVR Scheduler Started"
Write-Host "Parameter Search: 12:00 UTC"
Write-Host "Paper Trading: 14:00-19:00 UTC"
Write-Host ""

while ($true) {

    $now = (Get-Date).ToUniversalTime()
    $date = $now.ToString("yyyy-MM-dd")
    $hour = $now.Hour
    $minute = $now.Minute
    $day = $now.DayOfWeek

    # Monday-Friday only
    if ($day -ne "Saturday" -and $day -ne "Sunday") {

        # Daily Parameter Search at 12:00 UTC
        if (
            $hour -eq 12 -and
            $minute -eq 0 -and
            $lastSearchDate -ne $date
        ) {
            Write-Host "[$now] Starting Daily Parameter Search"

            Set-Location $projectPath
            python3 mlpHyperparameterSearch.py

            $lastSearchDate = $date

            Write-Host "[$now] Parameter Search Completed"
        }


        # Hourly Paper Trading
        # 14:00, 15:00, 16:00, 17:00, 18:00, 19:00 UTC

        $tradeHours = @(14, 15, 16, 17, 18, 19)

        $currentTradeTime = "$date-$hour"

        if (
            $tradeHours -contains $hour -and
            $minute -eq 0 -and
            $lastTradeTime -ne $currentTradeTime
        ) {
            Write-Host "[$now] Starting Hourly Paper Trading"

            Set-Location $projectPath
            python3 hourlyTrade.py

            $lastTradeTime = $currentTradeTime

            Write-Host "[$now] Paper Trading Completed"
        }
    }

    Start-Sleep -Seconds 20
}