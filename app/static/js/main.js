document.addEventListener("DOMContentLoaded", function () {
    
    // --- ANIMATED CHART DEFAULTS ---
    Chart.defaults.color = '#64748b'; 
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.borderColor = 'transparent';

    const churnChartCanvas = document.getElementById('churnChart');
    
    if (churnChartCanvas) {
        const churnCount = parseInt(churnChartCanvas.dataset.churn);
        const safeCount = parseInt(churnChartCanvas.dataset.safe);

        new Chart(churnChartCanvas, {
            type: 'doughnut',
            data: {
                labels: ['Risk', 'Safe'],
                datasets: [{
                    data: [churnCount, safeCount],
                    backgroundColor: [
                        '#ef4444', // Red
                        '#10b981'  // Green
                    ],
                    borderWidth: 0,
                    hoverOffset: 15,
                    borderRadius: 5 // Rounded edges on the donut segments
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    animateScale: true,
                    animateRotate: true,
                    duration: 1500, // Slow, premium animation
                    easing: 'easeOutQuart'
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 25,
                            font: { size: 12, weight: '600' }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: false
                    }
                },
                cutout: '80%' // Very thin, modern ring
            }
        });
    }
});