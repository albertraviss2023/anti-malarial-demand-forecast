document.addEventListener('DOMContentLoaded', () => {
    const { DateTime } = luxon;
    let dashboardData = {};
    const skylineShades = ['#2A4F97', '#468FAF', '#91C8E4', '#C9E4FF'];
    let currentCharts = [];
    let map, markers = [];

    // Status thresholds
    const MALARIA_STATUS_THRESHOLDS = {
        HIGH: 5000,
        MEDIUM: 1500
    };

    // District coordinates
    const districtCoords = {
        'Abim': [2.7069, 33.660],
        'Kampala': [0.3476, 32.5825],
        'Gulu': [2.7746, 32.2980],
        'Mbarara': [-0.6057, 30.6485],
        'Jinja': [0.4244, 33.2041],
        'Mbale': [1.0647, 34.1796],
        'Arua': [3.0201, 30.9110],
        'Lira': [2.2350, 32.9099],
        'Masaka': [-0.3269, 31.7533],
        'Entebbe': [0.0644, 32.4465],
        'Fort Portal': [0.6933, 30.2666],
        'Soroti': [1.7156, 33.6091],
        'Tororo': [0.6933, 34.1809],
        'Kabale': [-1.2491, 29.9899],
        'Mityana': [0.4170, 32.0228],
        'Hoima': [1.4356, 31.3436],
        'Masindi': [1.6748, 31.7150],
        'Kasese': [0.1833, 30.0833],
        'Bushenyi': [-0.5350, 30.1856],
        'Iganga': [0.6092, 33.4686]
    };

    // Helper functions
    function formatNumber(value, decimals = 2) {
        return value === undefined || value === null || isNaN(value) ? 'N/A' : Number(value).toFixed(decimals);
    }

    function showError(message) {
        const alertElement = document.getElementById('alert');
        const alertMessage = document.getElementById('alert-message');
        if (alertElement && alertMessage) {
            alertMessage.textContent = message;
            alertElement.style.display = 'block';
        } else {
            console.error('Error: Alert elements not found');
            alert(message);
        }
    }

    function safeSetInnerHTML(id, content) {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = content;
        } else {
            console.error(`Element #${id} not found`);
        }
    }

    async function fetchWithLogging(url) {
        console.log(`Fetching: ${url}`);
        try {
            const response = await fetch(url);
            console.log(`Response status for ${url}: ${response.status}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            console.log(`Data received from ${url}:`, data);
            return data;
        } catch (error) {
            console.error(`Fetch error for ${url}:`, error);
            showError(`Failed to load data: ${error.message}`);
            throw error;
        }
    }

    function initMap() {
        const mapElement = document.getElementById('map');
        if (!mapElement) {
            console.error('Map element not found');
            return false;
        }

        try {
            // Ensure Leaflet is loaded
            if (!L) {
                throw new Error('Leaflet library not loaded');
            }

            // Initialize map centered on Uganda
            map = L.map('map', {
                preferCanvas: true,
                zoomControl: false
            }).setView([1.3733, 32.2903], 7);

            // Add tile layer
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom: 18
            }).addTo(map);

            // Add custom zoom control
            L.control.zoom({
                position: 'topright'
            }).addTo(map);

            // Show map container
            const mapContainer = document.getElementById('mapAndChartContainer');
            if (mapContainer) {
                mapContainer.style.display = 'block';
            }

            return true;
        } catch (err) {
            console.error('Map initialization failed:', err);
            showError('Failed to initialize map. Please try again.');
            return false;
        }
    }

    function updateMapWithData(data) {
        if (!map || !data) {
            console.warn('Map or data not available for update');
            return;
        }

        // Clear existing markers
        markers.forEach(marker => map.removeLayer(marker));
        markers = [];

        const month = document.getElementById('monthFilter')?.value || 'May_2025';
        const statusFilter = document.getElementById('statusFilter')?.value || 'all';
        const casesRange = parseInt(document.getElementById('casesRange')?.value) || 10000;

        // Add markers for each district
        data.forEach(district => {
            const cases = district.cases?.[month.split('_')[0]] || 0;

            // Determine status based on thresholds
            let status;
            if (cases >= MALARIA_STATUS_THRESHOLDS.HIGH) {
                status = 'high';
            } else if (cases >= MALARIA_STATUS_THRESHOLDS.MEDIUM) {
                status = 'medium';
            } else {
                status = 'low';
            }

            // Skip if doesn't match filter
            if (statusFilter !== 'all' && status !== statusFilter) return;
            if (cases > casesRange) return;

            const coords = districtCoords[district.district];
            if (!coords) {
                console.warn(`No coordinates found for district: ${district.district}`);
                return;
            }

            const color = status === 'high' ? '#e63946' : status === 'medium' ? '#ff7d00' : '#00a86b';

            const marker = L.circleMarker(coords, {
                radius: Math.min(20, Math.max(5, cases / 200)),
                fillColor: color,
                color: '#fff',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8,
                className: `${status}-marker`
            }).addTo(map);

            if (status === 'high') {
                marker.setStyle({
                    className: 'high-marker animate__animated animate__pulse animate__infinite'
                });
            }

            marker.bindPopup(`
                <b>${district.district}</b><br>
                Status: <span class="${status}">${status.toUpperCase()}</span><br>
                Cases: ${formatNumber(cases)}<br>
                Month: ${month.replace('_', ' ')}
            `);

            markers.push(marker);
        });

        // Fit map to bounds if markers exist
        if (markers.length > 0) {
            const group = new L.featureGroup(markers);
            map.fitBounds(group.getBounds().pad(0.2));
        }

        // Update stats chart
        updateStatsChart(data, month);
    }

    function updateStatsChart(data, month) {
        const statsChart = document.getElementById('statsChart');
        if (!statsChart || !data) return;

        const monthKey = month || 'May_2025';
        const monthName = monthKey.replace('_', ' ');

        const sortedData = [...data].sort((a, b) =>
            (b.cases?.[monthKey.split('_')[0]] || 0) - (a.cases?.[monthKey.split('_')[0]] || 0)
        ).slice(0, 15);

        const chartData = [{
            x: sortedData.map(d => d.district),
            y: sortedData.map(d => d.cases?.[monthKey.split('_')[0]] || 0),
            type: 'bar',
            marker: {
                color: sortedData.map(d => {
                    const cases = d.cases?.[monthKey.split('_')[0]] || 0;
                    return cases >= MALARIA_STATUS_THRESHOLDS.HIGH ? '#e63946' :
                           cases >= MALARIA_STATUS_THRESHOLDS.MEDIUM ? '#ff7d00' : '#00a86b';
                })
            }
        }];

        const layout = {
            title: `Top Districts - ${monthName}`,
            xaxis: {
                title: 'District',
                tickangle: -45,
                type: 'category'
            },
            yaxis: {
                title: 'Cases',
                rangemode: 'tozero'
            },
            height: 250,
            margin: { t: 50, b: 100, l: 60, r: 40 }
        };

        Plotly.newPlot('statsChart', chartData, layout);
    }

    function filterMarkers() {
        const status = document.getElementById('statusFilter')?.value || 'all';
        markers.forEach(marker => {
            const markerStatus = marker.options.className.split('-')[0];
            if (status === 'all' || markerStatus === status) {
                marker.setStyle({ opacity: 1, fillOpacity: 0.8 });
            } else {
                marker.setStyle({ opacity: 0.2, fillOpacity: 0.2 });
            }
        });
    }

    async function loadDashboardData() {
        const loadingSpinner = document.getElementById('loadingSpinner');
        if (loadingSpinner) loadingSpinner.style.display = 'flex';

        try {
            const [inputData, testMetrics, predictions, malariaMaes, malariaPredictions, historicalMalaria] = await Promise.all([
                fetchWithLogging('/api/input-data'),
                fetchWithLogging('/api/test-metrics'),
                fetchWithLogging('/api/predictions'),
                fetchWithLogging('/api/malaria-maes'),
                fetchWithLogging('/api/malaria-predictions'),
                fetchWithLogging('/api/historical-malaria')
            ]);

            dashboardData = {
                inputData,
                testMetrics,
                predictions,
                malariaMaes,
                malariaPredictions,
                historicalMalaria
            };

            console.log('All data loaded successfully');
            updateDashboard();
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            showError('Unable to load dashboard data. Please try refreshing.');
        } finally {
            if (loadingSpinner) loadingSpinner.style.display = 'none';
        }
    }

    function updateDashboard() {
        const {
            inputData,
            testMetrics,
            predictions,
            malariaMaes,
            malariaPredictions,
            historicalMalaria
        } = dashboardData;

        if (!inputData || !testMetrics || !predictions) {
            showError('Incomplete data received.');
            return;
        }

        updateMetrics(inputData);
        updateOverviewTab(inputData, predictions);
        updateInputDataTab(inputData);
        updatePredictionsTab(predictions);
        updateComparisonTab(testMetrics, predictions);

        if (malariaPredictions && malariaMaes && historicalMalaria) {
            updateMalariaOverviewTab(malariaPredictions, malariaMaes, historicalMalaria);
        }

        safeSetInnerHTML('currentDate', DateTime.now().toFormat('MMMM dd, yyyy'));
    }

    function updateMetrics(inputData) {
        safeSetInnerHTML('currentDemand', `<span class="number-tick">${formatNumber(inputData.stats?.current_demand)}</span>`);
        safeSetInnerHTML('avgTemp', `<span class="number-tick">${formatNumber((inputData.stats?.max_temp_avg + inputData.stats?.min_temp_avg) / 2)}°C</span>`);
        safeSetInnerHTML('precipTotal', `<span class="number-tick">${formatNumber(inputData.stats?.precip_total)} mm</span>`);
        safeSetInnerHTML('humidityAvg', `<span class="number-tick">${formatNumber(inputData.stats?.humidity_avg)}%</span>`);

        safeSetInnerHTML('demandTrend', inputData.stats?.current_demand > 0.8 ? 'High' : 'Stable');
        safeSetInnerHTML('tempTrend', inputData.stats?.max_temp_avg > 25 ? 'Warm' : 'Normal');
        safeSetInnerHTML('precipTrend', inputData.stats?.precip_total > 10000 ? 'High' : 'Normal');
        safeSetInnerHTML('humidityTrend', inputData.stats?.humidity_avg > 60 ? 'High' : 'Normal');
    }

    function updateOverviewTab(inputData, predictions) {
        if (currentCharts.overviewChart) {
            currentCharts.overviewChart.destroy();
        }

        const overviewChart = document.getElementById('overviewChart');
        if (overviewChart) {
            const ctx = overviewChart.getContext('2d');
            currentCharts.overviewChart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Historical DDD Demand',
                            data: predictions.last_6_months?.map(d => ({
                                x: DateTime.fromISO(d.date).toJSDate(),
                                y: d.ddd_demand
                            })) || [],
                            borderColor: '#355070',
                            backgroundColor: 'rgba(53, 80, 112, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Forecasted DDD Demand',
                            data: [
                                { x: DateTime.fromFormat('April 2025', 'MMMM yyyy').toJSDate(), y: predictions.predictions?.[predictions.best_model]?.['April 2025'] },
                                { x: DateTime.fromFormat('May 2025', 'MMMM yyyy').toJSDate(), y: predictions.predictions?.[predictions.best_model]?.['May 2025'] },
                                { x: DateTime.fromFormat('June 2025', 'MMMM yyyy').toJSDate(), y: predictions.predictions?.[predictions.best_model]?.['June 2025'] }
                            ],
                            borderColor: '#E56B6F',
                            backgroundColor: 'rgba(229, 107, 111, 0.1)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { usePointStyle: true, padding: 20 }
                        },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'month', displayFormats: { month: 'MMM yyyy' } },
                            title: { display: true, text: 'Date' },
                            grid: { display: false }
                        },
                        y: {
                            title: { display: true, text: 'DDD Demand' },
                            beginAtZero: true
                        }
                    },
                    interaction: { intersect: false, mode: 'nearest' }
                }
            });
        }

        if (currentCharts.riskChart) {
            currentCharts.riskChart.destroy();
        }

        const riskChart = document.getElementById('riskChart');
        if (riskChart) {
            const ctx = riskChart.getContext('2d');
            currentCharts.riskChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Precipitation', 'Temperature', 'Humidity', 'Sunshine'],
                    datasets: [{
                        label: 'Risk Score',
                        data: [
                            inputData.stats?.precip_total / 500 || 0,
                            ((inputData.stats?.max_temp_avg + inputData.stats?.min_temp_avg) / 2) || 0,
                            inputData.stats?.humidity_avg || 0,
                            inputData.stats?.sunshine_total / 1000 || 0
                        ],
                        backgroundColor: 'rgba(0, 168, 107, 0.2)',
                        borderColor: '#00a86b',
                        borderWidth: 2,
                        pointBackgroundColor: '#00a86b',
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        r: {
                            angleLines: { display: true, color: 'rgba(0, 0, 0, 0.1)' },
                            suggestedMin: 0,
                            suggestedMax: 100,
                            ticks: { stepSize: 20 }
                        }
                    }
                }
            });
        }

        safeSetInnerHTML('precipRisk', formatNumber(inputData.stats?.precip_total / 500));
        safeSetInnerHTML('tempRisk', formatNumber((inputData.stats?.max_temp_avg + inputData.stats?.min_temp_avg) / 2));
        safeSetInnerHTML('bestModelOverview', predictions.best_model || 'N/A');

        if (predictions.predictions?.[predictions.best_model]) {
            const preds = predictions.predictions[predictions.best_model];
            let peakMonth = '';
            let peakValue = 0;

            Object.entries(preds).forEach(([month, value]) => {
                if (value > peakValue) {
                    peakValue = value;
                    peakMonth = month;
                }
            });

            safeSetInnerHTML('peakDemandMonth', `${peakMonth} (${formatNumber(peakValue)})`);
        }
    }

    function updateInputDataTab(inputData) {
        const inputDataBody = document.getElementById('inputDataBody');
        if (inputDataBody) {
            inputDataBody.innerHTML = inputData.data?.map(row => `
                <tr class="animated-cell">
                    <td>${DateTime.fromISO(row.date).toFormat('MMM yyyy')}</td>
                    <td>${formatNumber(row.avg_temp_max)}</td>
                    <td>${formatNumber(row.avg_temp_min)}</td>
                    <td>${formatNumber(row.avg_humidity)}</td>
                    <td>${formatNumber(row.total_precipitation)}</td>
                    <td>${formatNumber(row.total_sunshine_hours)}</td>
                    <td>${formatNumber(row.ddd_demand)}</td>
                </tr>
            `).join('') || '<tr><td colspan="7">No data available</td></tr>';
        }

        if (currentCharts.environmentChart) {
            currentCharts.environmentChart.destroy();
        }

        const environmentChart = document.getElementById('environmentChart');
        if (environmentChart) {
            const ctx = environmentChart.getContext('2d');
            currentCharts.environmentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Temperature (°C)',
                            data: inputData.data?.map(d => ({
                                x: DateTime.fromISO(d.date).toJSDate(),
                                y: (d.avg_temp_max + d.avg_temp_min) / 2
                            })) || [],
                            borderColor: '#ff7d00',
                            backgroundColor: 'rgba(255, 125, 0, 0.1)',
                            borderWidth: 2,
                            yAxisID: 'y',
                            tension: 0.3
                        },
                        {
                            label: 'Precipitation (mm/100)',
                            data: inputData.data?.map(d => ({
                                x: DateTime.fromISO(d.date).toJSDate(),
                                y: d.total_precipitation / 100
                            })) || [],
                            borderColor: '#1a5cb8',
                            backgroundColor: 'rgba(26, 92, 184, 0.1)',
                            borderWidth: 2,
                            yAxisID: 'y1',
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top', labels: { usePointStyle: true, padding: 20 } },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'month', displayFormats: { month: 'MMM yyyy' } },
                            title: { display: true, text: 'Date' },
                            grid: { display: false }
                        },
                        y: {
                            title: { display: true, text: 'Temperature (°C)' },
                            position: 'left'
                        },
                        y1: {
                            title: { display: true, text: 'Precipitation (mm/100)' },
                            position: 'right',
                            grid: { drawOnChartArea: false }
                        }
                    },
                    interaction: { intersect: false, mode: 'nearest' }
                }
            });
        }

        safeSetInnerHTML('demandCorrelation', inputData.stats?.precip_total > 10000 ? 'High (Precipitation)' : 'Moderate');
    }

    function updatePredictionsTab(predictions) {
        safeSetInnerHTML('bestModelName', predictions.best_model || 'N/A');
        const bestPreds = predictions.predictions?.[predictions.best_model] || {};

        safeSetInnerHTML('aprForecast', formatNumber(bestPreds['April 2025']));
        safeSetInnerHTML('mayForecast', formatNumber(bestPreds['May 2025']));
        safeSetInnerHTML('junForecast', formatNumber(bestPreds['June 2025']));
        safeSetInnerHTML('totalForecast', formatNumber(
            (bestPreds['April 2025'] || 0) +
            (bestPreds['May 2025'] || 0) +
            (bestPreds['June 2025'] || 0)
        ));

        if (currentCharts.forecastChart) {
            currentCharts.forecastChart.destroy();
        }

        const forecastChart = document.getElementById('forecastChart');
        if (forecastChart) {
            const ctx = forecastChart.getContext('2d');
            currentCharts.forecastChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['April 2025', 'May 2025', 'June 2025'],
                    datasets: [{
                        label: predictions.best_model || 'Best Model',
                        data: [
                            bestPreds['April 2025'],
                            bestPreds['May 2025'],
                            bestPreds['June 2025']
                        ],
                        backgroundColor: '#91C8E4',
                        borderColor: '#468FAF',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: ${context.raw}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'DDD Demand' }
                        }
                    }
                }
            });
        }
    }

    function updateComparisonTab(testMetrics, predictions) {
        if (currentCharts.maeChart) {
            currentCharts.maeChart.destroy();
        }

        const maeChart = document.getElementById('maeChart');
        if (maeChart) {
            const ctx = maeChart.getContext('2d');
            currentCharts.maeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: testMetrics?.map(m => m.model_name) || [],
                    datasets: [{
                        label: 'MAE',
                        data: testMetrics?.map(m => m.mae) || [],
                        backgroundColor: testMetrics?.map((_, i) => skylineShades[i % skylineShades.length]) || []
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `MAE: ${context.raw}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'MAE' }
                        }
                    }
                }
            });
        }

        if (currentCharts.predictionChart) {
            currentCharts.predictionChart.destroy();
        }

        const predictionChart = document.getElementById('predictionChart');
        if (predictionChart) {
            const ctx = predictionChart.getContext('2d');
            currentCharts.predictionChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['April 2025', 'May 2025', 'June 2025'],
                    datasets: predictions.model_names?.map((model, i) => ({
                        label: model,
                        data: [
                            predictions.predictions?.[model]?.['April 2025'] || 0,
                            predictions.predictions?.[model]?.['May 2025'] || 0,
                            predictions.predictions?.[model]?.['June 2025'] || 0
                        ],
                        backgroundColor: skylineShades[i % skylineShades.length],
                        borderColor: '#2A4F97',
                        borderWidth: 1
                    })) || []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'DDD Demand' }
                        },
                        x: { stacked: false }
                    },
                    interaction: { intersect: false, mode: 'index' }
                }
            });
        }

        const comparisonTableHead = document.getElementById('comparisonTableHead');
        const comparisonTableBody = document.getElementById('comparisonTableBody');
        if (comparisonTableHead && comparisonTableBody) {
            comparisonTableHead.innerHTML = `
                <tr>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>Apr 2025</th>
                    <th>May 2025</th>
                    <th>Jun 2025</th>
                </tr>
            `;

            comparisonTableBody.innerHTML = testMetrics?.map(m => `
                <tr data-model="${m.model_name}" class="animated-cell">
                    <td>${m.model_name}</td>
                    <td class="mae-cell">${formatNumber(m.mae, 4)}</td>
                    <td>${formatNumber(predictions.predictions?.[m.model_name]?.['April 2025'])}</td>
                    <td>${formatNumber(predictions.predictions?.[m.model_name]?.['May 2025'])}</td>
                    <td>${formatNumber(predictions.predictions?.[m.model_name]?.['June 2025'])}</td>
                </tr>
            `).join('') || '<tr><td colspan="5">No data available</td></tr>';
        }
    }

    function updateMalariaOverviewTab(malariaPredictions, malariaMaes, historicalMalaria) {
        // Update status table
        const statusTableBody = document.getElementById('statusTableBody');
        if (statusTableBody && malariaPredictions?.data) {
            const statusCounts = { high: [0, 0, 0], medium: [0, 0, 0], low: [0, 0, 0] };

            malariaPredictions.data.forEach(d => {
                ['May', 'June', 'July'].forEach((month, i) => {
                    const cases = d.cases?.[month] || 0;
                    let status;
                    if (cases >= MALARIA_STATUS_THRESHOLDS.HIGH) {
                        status = 'high';
                    } else if (cases >= MALARIA_STATUS_THRESHOLDS.MEDIUM) {
                        status = 'medium';
                    } else {
                        status = 'low';
                    }
                    statusCounts[status][i]++;
                });
            });

            statusTableBody.innerHTML = `
                <tr><td>High</td><td>${statusCounts.high[0]}</td><td>${statusCounts.high[1]}</td><td>${statusCounts.high[2]}</td></tr>
                <tr><td>Medium</td><td>${statusCounts.medium[0]}</td><td>${statusCounts.medium[1]}</td><td>${statusCounts.medium[2]}</td></tr>
                <tr><td>Low</td><td>${statusCounts.low[0]}</td><td>${statusCounts.low[1]}</td><td>${statusCounts.low[2]}</td></tr>
            `;
        }

        // Update model performance
        const modelPerformance = document.getElementById('modelPerformance');
        if (modelPerformance && malariaPredictions?.best_model && malariaMaes) {
            const bestModelData = malariaMaes.find(m => m.model_name === malariaPredictions.best_model);

            safeSetInnerHTML('bestMalariaModel', malariaPredictions.best_model || 'N/A');
            safeSetInnerHTML('bestMalariaMAE', formatNumber(bestModelData?.mae_sum || 0));

            Plotly.newPlot('modelPerformance', [{
                x: ['MAE Sum', 't+1 MAE', 't+2 MAE', 't+3 MAE'],
                y: [
                    bestModelData?.mae_sum || 0,
                    bestModelData?.['t+1_mae'] || 0,
                    bestModelData?.['t+2_mae'] || 0,
                    bestModelData?.['t+3_mae'] || 0
                ],
                type: 'bar',
                marker: { color: '#1a5cb8' }
            }], {
                title: 'Model Error Metrics',
                xaxis: { title: 'Metric' },
                yaxis: { title: 'MAE Value' },
                height: 250,
                margin: { t: 50, b: 60, l: 60, r: 40 }
            });
        }

        // Update historical trends
        const historicalTrend = document.getElementById('historicalTrend');
        if (historicalTrend && historicalMalaria?.data) {
            // Process historical data
            const monthlyData = {};
            historicalMalaria.data.forEach(item => {
                if (!item.date || !item.cases) return; // Skip invalid entries
                const date = DateTime.fromISO(item.date);
                if (!date.isValid) return;
                const monthYear = date.toFormat('MMM yyyy');
                monthlyData[monthYear] = (monthlyData[monthYear] || 0) + (item.cases || 0);
            });

            // Get last 6 months
            const last6Months = Object.entries(monthlyData).sort((a, b) =>
                DateTime.fromFormat(a[0], 'MMM yyyy') - DateTime.fromFormat(b[0], 'MMM yyyy')
            ).slice(-6);

            // Find peak month
            let peakMonth = '';
            let peakCases = 0;
            let totalCases = 0;

            last6Months.forEach(([month, cases]) => {
                totalCases += cases;
                if (cases > peakCases) {
                    peakCases = cases;
                    peakMonth = month;
                }
            });

            safeSetInnerHTML('peakMonth', peakMonth ? `${peakMonth} (${formatNumber(peakCases)} cases)` : 'N/A');
            safeSetInnerHTML('totalCases', formatNumber(totalCases));

            // Create chart
            if (last6Months.length > 0) {
                Plotly.newPlot('historicalTrend', [{
                    x: last6Months.map(([month]) => month),
                    y: last6Months.map(([_, cases]) => cases),
                    type: 'line+bar',
                    marker: { color: '#00a86b' },
                    line: { color: '#00a86b' }
                }], {
                    title: 'Historical Malaria Cases (Last 6 Months)',
                    xaxis: { title: 'Month', type: 'category' },
                    yaxis: { title: 'Cases', rangemode: 'tozero' },
                    height: 250,
                    margin: { t: 50, b: 60, l: 60, r: 40 }
                });
            } else {
                safeSetInnerHTML('historicalTrend', '<p>No historical data available</p>');
            }
        } else {
            safeSetInnerHTML('historicalTrend', '<p>No historical data available</p>');
        }
    }

    function toggleTheme() {
        const isDark = document.body.dataset.theme === 'dark';
        document.body.setAttribute('data-theme', isDark ? 'light' : 'dark');
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.innerHTML = `<i class="fas ${isDark ? 'fa-moon' : 'fa-sun'}"></i>`;
            themeToggle.setAttribute('title', `Toggle ${isDark ? 'Light' : 'Dark'} Mode`);
        }

        Object.values(currentCharts).forEach(chart => {
            if (chart && typeof chart.update === 'function') {
                chart.update();
            }
        });

        if (typeof Plotly !== 'undefined') {
            Plotly.relayout('modelPerformance', {});
            Plotly.relayout('historicalTrend', {});
            Plotly.relayout('statsChart', {});
        }
    }

    // Event listeners
    document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);

    document.getElementById('refreshData')?.addEventListener('click', e => {
        e.preventDefault();
        loadDashboardData();
    });

    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            if (e.target.id === 'advanced-map-tab' && !map) {
                if (initMap() && dashboardData.malariaPredictions?.data) {
                    updateMapWithData(dashboardData.malariaPredictions.data);
                }
            }
        });
    });

    // Add event listeners for filters
    document.getElementById('monthFilter')?.addEventListener('change', window.updateMap);
    document.getElementById('statusFilter')?.addEventListener('change', window.filterMarkers);
    document.getElementById('casesRange')?.addEventListener('input', window.updateMap);

    // Global functions for map controls
    window.updateMap = function() {
        const month = document.getElementById('monthFilter')?.value || 'May_2025';
        if (dashboardData.malariaPredictions?.data) {
            updateMapWithData(dashboardData.malariaPredictions.data);
        }
    };

    window.filterMarkers = filterMarkers;

    // Initialize dashboard
    loadDashboardData();
});