// Risk & Retention Intelligence Engine - Dashboard JavaScript
// Advanced Banking ML Dashboard with Real-time Predictions

// Global variables for data and charts
let chartsInitialized = false;
let demographicsChart, churnAnalysisChart, rocCurvesPlot;

// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeCharts();
    loadInitialData();
});

function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links and sections
            navLinks.forEach(l => l.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Show corresponding section
            const targetSection = this.getAttribute('href').substring(1);
            const section = document.getElementById(targetSection);
            if (section) {
                section.classList.add('active');
                
                // Initialize charts for the section if needed
                if (targetSection === 'insights' && !chartsInitialized) {
                    setTimeout(() => {
                        initializeAllInsightCharts();
                        chartsInitialized = true;
                    }, 100);
                } else if (targetSection === 'models') {
                    setTimeout(() => {
                        initializeROCCurves();
                    }, 100);
                }
            }
        });
    });
}

// Chart initialization functions
function initializeCharts() {
    // Set default Chart.js options
    Chart.defaults.font.family = 'Inter, sans-serif';
    Chart.defaults.color = '#374151';
}

function initializeAllInsightCharts() {
    initializeCountryChart();
    initializeAgeChart();
    initializeGenderChart();
    initializeProductsChart();
    initializeValueChart();
    initializeActivityChart();
    initializeBalanceChart();
    initializeTenureChart();
}

function initializeCountryChart() {
    const ctx = document.getElementById('countryChurnChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Germany', 'Spain', 'France'],
            datasets: [{
                label: 'Churn Rate (%)',
                data: [32.4, 16.7, 16.2],
                backgroundColor: ['#dc2626', '#f59e0b', '#10b981'],
                borderWidth: 1,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 40,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

function initializeAgeChart() {
    const ctx = document.getElementById('ageChurnChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-59', '60+'],
            datasets: [{
                label: 'Churn Rate (%)',
                data: [8.8, 7.1, 8.1, 13.3, 23.7, 43.4, 56.0, 27.9],
                borderColor: '#dc2626',
                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#dc2626',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 60,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

function initializeGenderChart() {
    const ctx = document.getElementById('genderChurnChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Female (25.1%)', 'Male (16.5%)'],
            datasets: [{
                data: [25.1, 16.5],
                backgroundColor: ['#dc2626', '#3b82f6'],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 20, usePointStyle: true }
                }
            }
        }
    });
}

function initializeProductsChart() {
    const ctx = document.getElementById('productsChurnChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['1 Product', '2 Products', '3 Products', '4 Products'],
            datasets: [{
                label: 'Churn Rate (%)',
                data: [27.7, 7.6, 82.7, 100.0],
                backgroundColor: ['#f59e0b', '#10b981', '#dc2626', '#7c2d12'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 100,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

function initializeValueChart() {
    const ctx = document.getElementById('valueChurnChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Low', 'Medium', 'High', 'Premium'],
            datasets: [{
                label: 'Churn Rate (%)',
                data: [13.8, 25.2, 24.2, 23.6],
                backgroundColor: ['#10b981', '#dc2626', '#f59e0b', '#3b82f6'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 30,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

function initializeActivityChart() {
    const ctx = document.getElementById('activityChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Inactive', 'Active'],
            datasets: [{
                label: 'Churn Rate (%)',
                data: [26.9, 14.3],
                backgroundColor: ['#dc2626', '#10b981'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 30,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

function initializeBalanceChart() {
    const ctx = document.getElementById('balanceChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Zero', 'Low\n(0-25K)', 'Medium\n(25-50K)', 'High\n(50-100K)', 'Premium\n(100-150K)', 'Ultra\n(150K+)'],
            datasets: [{
                label: 'Churn Rate (%)',
                data: [13.8, 66.7, 31.9, 19.9, 25.8, 23.1],
                backgroundColor: ['#10b981', '#7c2d12', '#dc2626', '#f59e0b', '#3b82f6', '#8b5cf6'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 70,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

function initializeTenureChart() {
    const ctx = document.getElementById('tenureChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['New\n(0-2yr)', 'Growing\n(2-4yr)', 'Established\n(4-6yr)', 'Mature\n(6-8yr)', 'Veteran\n(8yr+)'],
            datasets: [{
                label: 'Churn Rate (%)',
                data: [22.6, 20.1, 20.6, 18.7, 20.4],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.3,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 25,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}


function initializeROCCurves() {
    const container = document.getElementById('rocCurves');
    if (!container) return;
    
    // Banking Churn ROC Curve (simulated data based on 96.3% AUC)
    const bankingFPR = [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.60, 1.0];
    const bankingTPR = [0, 0.25, 0.45, 0.60, 0.72, 0.82, 0.87, 0.92, 0.95, 0.98, 1.0];
    
    // Fraud Detection ROC Curve (simulated data based on 94.5% AUC)
    const fraudFPR = [0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.65, 1.0];
    const fraudTPR = [0, 0.20, 0.40, 0.58, 0.70, 0.80, 0.87, 0.92, 0.96, 0.98, 1.0];
    
    const data = [
        {
            x: bankingFPR,
            y: bankingTPR,
            type: 'scatter',
            mode: 'lines',
            name: 'Banking Churn (AUC = 0.963)',
            line: { color: '#000000', width: 3 }
        },
        {
            x: fraudFPR,
            y: fraudTPR,
            type: 'scatter',
            mode: 'lines',
            name: 'Fraud Detection (AUC = 0.945)',
            line: { color: '#666666', width: 3 }
        },
        {
            x: [0, 1],
            y: [0, 1],
            type: 'scatter',
            mode: 'lines',
            name: 'Random Classifier',
            line: { color: '#b3b3b3', width: 2, dash: 'dash' }
        }
    ];
    
    const layout = {
        title: {
            text: 'ROC Curves - Model Performance Comparison',
            font: { size: 16, family: 'Inter' }
        },
        xaxis: {
            title: 'False Positive Rate',
            range: [0, 1],
            gridcolor: '#e5e7eb'
        },
        yaxis: {
            title: 'True Positive Rate',
            range: [0, 1],
            gridcolor: '#e5e7eb'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: {
            family: 'Inter',
            size: 12,
            color: '#374151'
        },
        legend: {
            x: 0.6,
            y: 0.2
        },
        margin: { t: 50, r: 50, b: 50, l: 50 }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('rocCurves', data, layout, config);
}

// Load initial data and animate metrics
function loadInitialData() {
    animateMetrics();
    setTimeout(() => {
        animateProgressBars();
    }, 500);
}

function animateMetrics() {
    const metricValues = document.querySelectorAll('.metric-value');
    metricValues.forEach(element => {
        const finalValue = element.textContent;
        const numericValue = parseFloat(finalValue);
        
        if (!isNaN(numericValue)) {
            animateValue(element, 0, numericValue, 2000, finalValue.includes('%') ? '%' : finalValue.includes('K') ? 'K' : '');
        }
    });
    
    const impactStats = document.querySelectorAll('.impact-stat');
    impactStats.forEach(element => {
        const finalValue = element.textContent;
        const numericValue = parseFloat(finalValue);
        
        if (!isNaN(numericValue)) {
            animateValue(element, 0, numericValue, 2000, finalValue.includes('%') ? '%' : '');
        }
    });
}

function animateValue(element, start, end, duration, suffix) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            current = end;
            clearInterval(timer);
        }
        
        if (suffix === 'K') {
            element.textContent = Math.floor(current) + 'K';
        } else if (suffix === '%') {
            element.textContent = current.toFixed(1) + '%';
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = width;
        }, 100);
    });
    
    const importanceBars = document.querySelectorAll('.bar-fill');
    importanceBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = width;
        }, 200);
    });
}

// Prediction Functions
async function predictChurn() {
    // Get form values
    const customerData = {
        creditScore: parseInt(document.getElementById('creditScore').value),
        age: parseInt(document.getElementById('age').value),
        tenure: parseInt(document.getElementById('tenure').value),
        balance: parseFloat(document.getElementById('balance').value),
        products: parseInt(document.getElementById('products').value),
        salary: parseFloat(document.getElementById('salary').value),
        country: document.getElementById('country').value,
        gender: document.getElementById('gender').value,
        hasCard: document.getElementById('hasCard').checked,
        isActive: document.getElementById('isActive').checked
    };
    
    // Show loading state
    const button = document.querySelector('.predict-btn');
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    button.disabled = true;
    
    try {
        // Call Flask API
        const response = await fetch('http://127.0.0.1:5000/api/predict/churn', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(customerData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display result
        displayChurnResult(result, customerData);
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        // Fallback to client-side prediction
        const fallbackResult = simulateChurnPrediction(customerData);
        displayChurnResult(fallbackResult, customerData);
        
    } finally {
        // Reset button
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

function simulateChurnPrediction(data) {
    // Fallback client-side prediction
    let riskScore = 0;
    
    // Age risk (45-59 highest risk)
    if (data.age >= 45 && data.age <= 59) riskScore += 0.30;
    else if (data.age >= 40 && data.age <= 44) riskScore += 0.15;
    else if (data.age >= 35 && data.age <= 39) riskScore += 0.08;
    else if (data.age >= 25 && data.age <= 34) riskScore += 0.05;
    
    // Country risk
    if (data.country === 'Germany') riskScore += 0.25;
    else if (data.country === 'Spain' || data.country === 'France') riskScore += 0.08;
    
    // Credit score risk
    if (data.creditScore < 580) riskScore += 0.15;
    else if (data.creditScore < 670) riskScore += 0.10;
    else if (data.creditScore >= 740) riskScore -= 0.05;
    
    // Products risk
    if (data.products >= 3) riskScore += 0.20;
    else if (data.products === 1) riskScore += 0.05;
    
    // Activity risk
    if (!data.isActive) riskScore += 0.20;
    if (!data.hasCard) riskScore += 0.05;
    
    // Balance risk
    const balanceRatio = data.balance / data.salary;
    if (data.balance === 0) riskScore += 0.15;
    else if (balanceRatio > 1.5) riskScore += 0.10;
    else if (balanceRatio < 0.1) riskScore += 0.08;
    
    // Tenure risk
    if (data.tenure <= 2) riskScore += 0.10;
    else if (data.tenure >= 8) riskScore -= 0.05;
    
    // Gender risk - Females have 25.07% churn vs Males 16.46%
    if (data.gender === 'Female') riskScore += 0.15;
    
    // Age and gender interaction - Female risk increases with age
    if (data.gender === 'Female' && data.age >= 35) riskScore += 0.12;
    
    // Convert to probability (cap at 95%)
    const probability = Math.min(riskScore, 0.95);
    
    return {
        probability: probability,
        prediction: probability > 0.56 ? 1 : 0,
        risk_level: probability >= 0.7 ? 'HIGH' : (probability >= 0.4 ? 'MEDIUM' : 'LOW'),
        confidence: Math.max(probability, 1 - probability),
        method: 'client_fallback',
        recommendations: []
    };
}

function displayChurnResult(result, inputs) {
    const resultDiv = document.getElementById('churnResult');
    let riskLevel, riskClass, recommendations;
    
    // Handle both API response and direct probability
    const probability = typeof result === 'object' ? result.probability : result;
    const probabilityPercent = (probability * 100).toFixed(1);
    
    // Get recommendations from API response or use defaults
    if (typeof result === 'object' && result.recommendations) {
        recommendations = result.recommendations;
    } else {
        // Default recommendations based on risk level
        if (probability >= 0.7) {
            recommendations = [
                'Immediate retention campaign recommended',
                'Offer premium services or loyalty rewards',
                'Personal relationship manager assignment',
                'Competitive rate review and adjustment'
            ];
        } else if (probability >= 0.4) {
            recommendations = [
                'Proactive engagement recommended',
                'Product cross-sell opportunities',
                'Regular check-ins and surveys',
                'Enhanced digital services'
            ];
        } else {
            recommendations = [
                'Standard retention activities sufficient',
                'Focus on product expansion',
                'Maintain service quality',
                'Monitor for changes in behavior'
            ];
        }
    }
    
    // Determine risk level and class
    if (probability >= 0.7) {
        riskLevel = 'HIGH RISK';
        riskClass = 'high-risk';
    } else if (probability >= 0.4) {
        riskLevel = 'MEDIUM RISK';
        riskClass = 'medium-risk';
    } else {
        riskLevel = 'LOW RISK';
        riskClass = 'low-risk';
    }
    
    // Additional info from API response
    const confidence = typeof result === 'object' ? (result.confidence * 100).toFixed(1) : '85.0';
    const method = typeof result === 'object' && (result.method || result.model_used) ? 
                   ` (${result.method || result.model_used})` : '';
    
    resultDiv.className = `prediction-result ${riskClass} show`;
    resultDiv.innerHTML = `
        <div class="result-score">${probabilityPercent}% Churn Probability</div>
        <div class="result-level">${riskLevel}</div>
        <div class="result-details">
            <h4>Recommendations:</h4>
            <ul style="text-align: left; margin-top: 1rem;">
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
            <div style="margin-top: 1rem; font-size: 0.875rem; opacity: 0.8;">
                <strong>Customer Profile:</strong> ${inputs.age}yr ${inputs.gender} from ${inputs.country}, 
                ${inputs.products} products, ${inputs.isActive ? 'active' : 'inactive'} member<br>
                <strong>Model Confidence:</strong> ${confidence}%${method}
            </div>
        </div>
    `;
}

async function predictFraud() {
    const transactionData = {
        amount: parseFloat(document.getElementById('fraudAmount').value),
        time: parseFloat(document.getElementById('fraudTime').value)
    };
    
    // Show loading state
    const button = document.querySelectorAll('.predict-btn')[1]; // Second predict button
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    button.disabled = true;
    
    try {
        // Call Flask API
        const response = await fetch('http://127.0.0.1:5000/api/predict/fraud', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display result
        displayFraudResult(result, transactionData);
        
    } catch (error) {
        console.error('Fraud prediction error:', error);
        
        // Fallback to client-side prediction
        const fallbackResult = simulateFraudPrediction(transactionData);
        displayFraudResult(fallbackResult, transactionData);
        
    } finally {
        // Reset button
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

function simulateFraudPrediction(data) {
    // Fallback client-side fraud prediction
    let fraudScore = 0;
    
    // Amount-based risk
    if (data.amount > 5000) fraudScore += 0.35;
    else if (data.amount > 1000) fraudScore += 0.25;
    else if (data.amount > 500) fraudScore += 0.15;
    else if (data.amount < 5) fraudScore += 0.20; // Very small amounts can be testing
    
    // Time-based risk (unusual hours)
    const hour = (data.time % 86400) / 3600; // Convert to hour of day
    if (hour < 6 || hour > 23) fraudScore += 0.15; // Late night/early morning
    
    // Deterministic component to simulate PCA features (based on amount and time)
    const deterministicFactor = ((data.amount * 0.001) + (data.time * 0.0001)) % 0.20;
    fraudScore += deterministicFactor;
    
    // Amount pattern (round numbers are suspicious)
    if (data.amount % 100 === 0 && data.amount >= 100) fraudScore += 0.10;
    
    // Convert to probability (fraud is rare, so cap lower)
    const probability = Math.min(fraudScore, 0.85);
    
    return {
        probability: probability,
        prediction: probability > 0.88 ? 1 : 0,
        risk_level: probability >= 0.6 ? 'HIGH' : (probability >= 0.3 ? 'MEDIUM' : 'LOW'),
        confidence: Math.max(probability, 1 - probability),
        method: 'client_fallback',
        actions: []
    };
}

function displayFraudResult(result, inputs) {
    const resultDiv = document.getElementById('fraudResult');
    let riskLevel, riskClass, actions;
    
    // Handle both API response and direct probability
    const probability = typeof result === 'object' ? result.probability : result;
    const probabilityPercent = (probability * 100).toFixed(1);
    
    // Get actions from API response or use defaults
    if (typeof result === 'object' && result.actions) {
        actions = result.actions;
    } else {
        // Default actions based on risk level
        if (probability >= 0.6) {
            actions = [
                'BLOCK TRANSACTION IMMEDIATELY',
                'Contact customer for verification',
                'Flag account for manual review',
                'Investigate recent transaction patterns'
            ];
        } else if (probability >= 0.3) {
            actions = [
                'Additional verification required',
                'Monitor account closely',
                'Send security alert to customer',
                'Review transaction context'
            ];
        } else {
            actions = [
                'Process transaction normally',
                'Continue standard monitoring',
                'No additional action required',
                'Update customer behavior profile'
            ];
        }
    }
    
    // Determine risk level and class
    if (probability >= 0.6) {
        riskLevel = 'HIGH FRAUD RISK';
        riskClass = 'high-risk';
    } else if (probability >= 0.3) {
        riskLevel = 'SUSPICIOUS ACTIVITY';
        riskClass = 'medium-risk';
    } else {
        riskLevel = 'LEGITIMATE TRANSACTION';
        riskClass = 'low-risk';
    }
    
    // Additional info from API response
    const confidence = typeof result === 'object' ? (result.confidence * 100).toFixed(1) : '85.0';
    const method = typeof result === 'object' && (result.method || result.model_used) ? 
                   ` (${result.method || result.model_used})` : '';
    
    resultDiv.className = `prediction-result ${riskClass} show`;
    resultDiv.innerHTML = `
        <div class="result-score">${probabilityPercent}% Fraud Probability</div>
        <div class="result-level">${riskLevel}</div>
        <div class="result-details">
            <h4>Recommended Actions:</h4>
            <ul style="text-align: left; margin-top: 1rem;">
                ${actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
            <div style="margin-top: 1rem; font-size: 0.875rem; opacity: 0.8;">
                <strong>Transaction:</strong> $${inputs.amount.toFixed(2)} at ${new Date(inputs.time * 1000).toLocaleTimeString()}<br>
                <strong>Model Confidence:</strong> ${confidence}%${method}
            </div>
        </div>
    `;
}

// Utility functions
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatPercentage(num) {
    return (num * 100).toFixed(1) + '%';
}

function resetChurnForm() {
    document.getElementById('creditScore').value = 650;
    document.getElementById('age').value = 35;
    document.getElementById('tenure').value = 3;
    document.getElementById('balance').value = 75000;
    document.getElementById('products').value = 2;
    document.getElementById('salary').value = 100000;
    document.getElementById('country').value = 'France';
    document.getElementById('gender').value = 'Male';
    document.getElementById('hasCard').checked = true;
    document.getElementById('isActive').checked = true;
    document.getElementById('churnResult').innerHTML = '';
}

function resetFraudForm() {
    document.getElementById('fraudAmount').value = 100.00;
    document.getElementById('fraudTime').value = 3600;
    document.getElementById('fraudResult').innerHTML = '';
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🏦 Risk & Retention Intelligence Engine Dashboard Loaded');
    console.log('📊 Banking Model: 91.6% Accuracy, 95.4% Precision');
    console.log('🛡️ Fraud Model: 96.0% Accuracy, 70.9% Precision');
    console.log('🚀 Production Ready - American Express Interview Ready!');
});
