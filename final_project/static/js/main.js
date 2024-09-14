let genderChart;
let isRunning = false;
let videoFeed = document.getElementById('videoFeed');
let statusInterval;

document.getElementById('startButton').addEventListener('click', function() {
    if (!isRunning) {
        startSurveillance();
        this.textContent = 'Stop Surveillance';
    } else {
        stopSurveillance();
        this.textContent = 'Start Surveillance';
    }
    isRunning = !isRunning;
});

function startSurveillance() {
    videoFeed.src = "/video_feed";
    videoFeed.style.display = 'block';
    startStatusFetch();
}

function stopSurveillance() {
    videoFeed.src = "";
    videoFeed.style.display = 'none';
    clearInterval(statusInterval);
    resetCounts();
}

function resetCounts() {
    updateCounts(0, 0);
    updateAgeGroups({});
    updateAlerts([]);
}

function updateCounts(maleCount, femaleCount) {
    document.getElementById('maleCount').textContent = maleCount;
    document.getElementById('femaleCount').textContent = femaleCount;

    if (genderChart) {
        genderChart.data.datasets[0].data = [maleCount, femaleCount];
        genderChart.update();
    } else {
        const ctx = document.getElementById('genderChart').getContext('2d');
        genderChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Male', 'Female'],
                datasets: [{
                    data: [maleCount, femaleCount],
                    backgroundColor: ['#36a2eb', '#ff6384']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#e0e0e0'
                        }
                    }
                }
            }
        });
    }
}

function updateAgeGroups(ageGroups) {
    const ageGroupsList = document.getElementById('ageGroupsList');
    ageGroupsList.innerHTML = '';
    for (const [ageGroup, count] of Object.entries(ageGroups)) {
        const ageGroupElement = document.createElement('div');
        ageGroupElement.textContent = `${ageGroup}: ${count}`;
        ageGroupsList.appendChild(ageGroupElement);
    }
}

function updateAlerts(alerts) {
    const alertsList = document.getElementById('alertsList');
    alertsList.innerHTML = '';
    alerts.forEach(alert => {
        const alertElement = document.createElement('div');
        alertElement.className = 'alert';
        alertElement.textContent = alert;
        alertsList.appendChild(alertElement);
    });
}

function fetchStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            updateCounts(data.male_count, data.female_count);
            updateAgeGroups(data.age_groups);
            updateAlerts(data.alerts);
        })
        .catch(error => console.error('Error:', error));
}

function startStatusFetch() {
    fetchStatus(); // Fetch immediately
    statusInterval = setInterval(fetchStatus, 1000); // Then fetch every second
}

window.addEventListener('resize', function() {
    if (genderChart) {
        genderChart.resize();
    }
});

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    resetCounts();
});