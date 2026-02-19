import { Chart, registerables, type ChartConfiguration } from 'chart.js';
Chart.register(...registerables);

const COLORS = {
    risk: {
        low: '#22c55e',
        medium: '#f59e0b',
        high: '#ef4444',
        critical: '#dc2626'
    },
    accent: {
        blue: '#3b82f6',
        gold: '#fbbf24',
        slate: '#0f172a'
    },
    text: {
        primary: '#1e293b',
        secondary: '#64748b'
    }
};

export function createRiskDistributionChart(canvasId: string, data?: { labels: string[], data: number[] }) {
    const ctx = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!ctx) return;

    // Clear existing chart
    const existingChart = Chart.getChart(ctx);
    if (existingChart) existingChart.destroy();

    const config: ChartConfiguration<'doughnut'> = {
        type: 'doughnut',
        data: {
            labels: data?.labels || ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
            datasets: [{
                data: data?.data || [15234, 6421, 2105, 822],
                backgroundColor: [
                    COLORS.risk.low,
                    COLORS.risk.medium,
                    COLORS.risk.high,
                    COLORS.risk.critical
                ],
                borderWidth: 0,
                hoverOffset: 15,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: { family: 'Inter', size: 12, weight: 600 },
                        color: COLORS.text.secondary
                    }
                },
                tooltip: {
                    backgroundColor: COLORS.accent.slate,
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: { size: 14, weight: 700 },
                    bodyFont: { size: 13 },
                    displayColors: false
                }
            }
        }
    };

    return new Chart(ctx, config);
}

export function createRiskTrendChart(canvasId: string, data?: { labels: string[], datasets: any[] }) {
    const ctx = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!ctx) return;

    // Clear existing chart c
    const existingChart = Chart.getChart(ctx);
    if (existingChart) existingChart.destroy();

    const config: ChartConfiguration<'line'> = {
        type: 'line',
        data: {
            labels: data?.labels || ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'],
            datasets: data?.datasets || [
                {
                    label: 'Baseline (No Action)',
                    data: [6.2, 6.8, 7.2, 7.5, 7.8, 8.1, 8.4],
                    borderColor: 'rgba(239, 68, 68, 0.4)',
                    borderWidth: 2,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Active Intervention',
                    data: [6.2, 6.5, 6.3, 6.1, 5.8, 5.2, 4.2],
                    borderColor: COLORS.accent.blue,
                    backgroundColor: 'rgba(59, 130, 246, 0.08)',
                    borderWidth: 3,
                    pointRadius: 4,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: COLORS.accent.blue,
                    pointBorderWidth: 2,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        boxWidth: 8,
                        boxHeight: 8,
                        usePointStyle: true,
                        font: { family: 'Inter', size: 12, weight: 500 },
                        color: COLORS.text.secondary
                    }
                },
                tooltip: {
                    backgroundColor: COLORS.accent.slate,
                    padding: 12,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11 }, color: COLORS.text.secondary }
                },
                y: {
                    grid: { color: 'rgba(241, 245, 249, 1)', drawTicks: false },
                    border: { display: false },
                    ticks: {
                        font: { size: 11 },
                        color: COLORS.text.secondary,
                        callback: (v) => v + '%',
                        padding: 10
                    }
                }
            }
        }
    };

    return new Chart(ctx, config);
}
