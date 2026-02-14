document.addEventListener("DOMContentLoaded", function () {
    
    // --- CHART DEFAULTS FOR DARK MODE ---
    Chart.defaults.color = '#a0a0a0'; 
    Chart.defaults.borderColor = '#333';
    
    // Check if we are on the dashboard
    const ctx = document.getElementById('churnChart');
    
    if (ctx) {
        // Read data from HTML data attributes
        const churnCount = parseInt(ctx.dataset.churn);
        const safeCount = parseInt(ctx.dataset.safe);

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['High Risk', 'Safe'],
                datasets: [{
                    data: [churnCount, safeCount],
                    backgroundColor: [
                        '#cf6679', // Red/Pink
                        '#03dac6'  // Teal
                    ],
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%', // Thin modern ring
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            font: { size: 12 }
                        }
                    }
                }
            }
        });
    }
});