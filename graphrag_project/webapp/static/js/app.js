// webapp/static/js/app.js
// RAPTOR vs GraphRAG Comparison Tool JavaScript

const API_BASE = '';

// State
let isDocumentIndexed = false;
let comparisonChart = null;
let queryHistory = [];
let pendingFile = null;  // For PDF/DOC uploads

// DOM Elements
const elements = {
    documentInput: document.getElementById('documentInput'),
    queryInput: document.getElementById('queryInput'),
    topKSelect: document.getElementById('topKSelect'),
    indexBtn: document.getElementById('indexBtn'),
    compareBtn: document.getElementById('compareBtn'),
    loadSampleBtn: document.getElementById('loadSampleBtn'),
    loadSampleQueriesBtn: document.getElementById('loadSampleQueriesBtn'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    fileInput: document.getElementById('fileInput'),
    indexStatus: document.getElementById('indexStatus'),
    resultsSection: document.getElementById('resultsSection'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    toastContainer: document.getElementById('toastContainer'),
    sampleQueriesDropdown: document.getElementById('sampleQueriesDropdown'),
    sampleQueriesList: document.getElementById('sampleQueriesList'),
    historyList: document.getElementById('historyList'),
    
    // Stats
    raptorWins: document.getElementById('raptorWins'),
    graphragWins: document.getElementById('graphragWins'),
    
    // Results
    winnerBanner: document.getElementById('winnerBanner'),
    winnerName: document.getElementById('winnerName'),
    totalQueryTime: document.getElementById('totalQueryTime'),
    
    // RAPTOR metrics
    raptorAvgScore: document.getElementById('raptorAvgScore'),
    raptorMaxScore: document.getElementById('raptorMaxScore'),
    raptorNumResults: document.getElementById('raptorNumResults'),
    raptorQueryTime: document.getElementById('raptorQueryTime'),
    raptorCoverage: document.getElementById('raptorCoverage'),
    raptorBadge: document.getElementById('raptorBadge'),
    raptorResults: document.getElementById('raptorResults'),
    
    // GraphRAG metrics
    graphragAvgScore: document.getElementById('graphragAvgScore'),
    graphragMaxScore: document.getElementById('graphragMaxScore'),
    graphragNumResults: document.getElementById('graphragNumResults'),
    graphragQueryTime: document.getElementById('graphragQueryTime'),
    graphragCoverage: document.getElementById('graphragCoverage'),
    graphragBadge: document.getElementById('graphragBadge'),
    graphragResults: document.getElementById('graphragResults')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadStats();
    initializeChart();
});

// Event Listeners
function initializeEventListeners() {
    elements.indexBtn.addEventListener('click', indexDocument);
    elements.compareBtn.addEventListener('click', runComparison);
    elements.loadSampleBtn.addEventListener('click', loadSampleDocument);
    elements.loadSampleQueriesBtn.addEventListener('click', toggleSampleQueries);
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
    
    elements.fileInput.addEventListener('change', handleFileUpload);
    
    elements.queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') runComparison();
    });
    
    // Close sample queries dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.query-section')) {
            elements.sampleQueriesDropdown.classList.remove('active');
        }
    });
}

// Initialize Chart
function initializeChart() {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'RAPTOR',
                    data: [],
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgb(16, 185, 129)',
                    borderWidth: 2
                },
                {
                    label: 'GraphRAG',
                    data: [],
                    backgroundColor: 'rgba(245, 158, 11, 0.6)',
                    borderColor: 'rgb(245, 158, 11)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Score Comparison by Result Rank',
                    color: '#f8fafc',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    labels: { color: '#94a3b8' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' },
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// API Functions
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'API request failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Index Document
async function indexDocument() {
    const text = elements.documentInput.value.trim();
    
    // Check if we have a pending file upload (PDF/DOC)
    if (pendingFile) {
        showLoading(true);
        updateIndexStatus('indexing');
        
        try {
            const formData = new FormData();
            formData.append('file', pendingFile);
            
            const response = await fetch(`${API_BASE}/api/index`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to process file');
            }
            
            const result = await response.json();
            
            isDocumentIndexed = true;
            pendingFile = null;
            elements.documentInput.readOnly = false;
            elements.documentInput.value = `[Document indexed from file]\n\nRAPTOR: ${result.details.raptor.chunks} chunks\nGraphRAG: ${result.details.graphrag.entities} entities`;
            updateIndexStatus('indexed', result.details);
            showToast('Document indexed successfully!', 'success');
            loadStats();
            
        } catch (error) {
            updateIndexStatus('not-indexed');
            showToast(`Error: ${error.message}`, 'error');
        } finally {
            showLoading(false);
        }
        return;
    }
    
    // Regular text input
    if (!text || text.startsWith('[File ready to upload')) {
        showToast('Please enter or upload a document first', 'warning');
        return;
    }
    
    showLoading(true);
    updateIndexStatus('indexing');
    
    try {
        const result = await apiCall('/api/index', {
            method: 'POST',
            body: JSON.stringify({ text })
        });
        
        isDocumentIndexed = true;
        updateIndexStatus('indexed', result.details);
        showToast('Document indexed successfully!', 'success');
        loadStats();
        
    } catch (error) {
        updateIndexStatus('not-indexed');
        showToast(`Error indexing document: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Run Comparison
async function runComparison() {
    const query = elements.queryInput.value.trim();
    
    if (!query) {
        showToast('Please enter a query', 'warning');
        return;
    }
    
    if (!isDocumentIndexed) {
        showToast('Please index a document first', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const topK = parseInt(elements.topKSelect.value);
        const result = await apiCall('/api/query', {
            method: 'POST',
            body: JSON.stringify({ query, top_k: topK })
        });
        
        displayResults(result);
        loadStats();
        elements.queryInput.value = '';
        
    } catch (error) {
        showToast(`Error running comparison: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Display Results
function displayResults(result) {
    elements.resultsSection.style.display = 'block';
    elements.resultsSection.classList.add('fade-in');
    
    // Update winner banner
    const winner = result.comparison.overall_winner;
    elements.winnerName.textContent = winner;
    elements.winnerBanner.className = `winner-banner ${winner.toLowerCase()}`;
    
    // Update metrics
    updateMetrics('raptor', result.raptor.metrics, winner === 'RAPTOR');
    updateMetrics('graphrag', result.graphrag.metrics, winner === 'GraphRAG');
    
    // Update total query time
    const totalTime = result.raptor.metrics.query_time_ms + result.graphrag.metrics.query_time_ms;
    elements.totalQueryTime.textContent = `Total: ${totalTime.toFixed(0)}ms`;
    
    // Update results lists
    displayResultsList('raptor', result.raptor.results);
    displayResultsList('graphrag', result.graphrag.results);
    
    // Update chart
    updateChart(result);
    
    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Update Metrics
function updateMetrics(system, metrics, isWinner) {
    const prefix = system;
    document.getElementById(`${prefix}AvgScore`).textContent = metrics.avg_score.toFixed(4);
    document.getElementById(`${prefix}MaxScore`).textContent = metrics.max_score.toFixed(4);
    document.getElementById(`${prefix}NumResults`).textContent = metrics.num_results;
    document.getElementById(`${prefix}QueryTime`).textContent = `${metrics.query_time_ms.toFixed(0)}ms`;
    document.getElementById(`${prefix}Coverage`).textContent = `${(metrics.coverage * 100).toFixed(1)}%`;
    
    const badge = document.getElementById(`${prefix}Badge`);
    if (isWinner) {
        badge.textContent = 'WINNER';
        badge.className = 'card-badge winner';
    } else {
        badge.textContent = 'RUNNER-UP';
        badge.className = 'card-badge loser';
    }
}

// Display Results List
function displayResultsList(system, results) {
    const container = document.getElementById(`${system}Results`);
    
    if (!results || results.length === 0) {
        container.innerHTML = '<p class="no-results">No results found</p>';
        return;
    }
    
    container.innerHTML = results.map((result, index) => `
        <div class="result-card slide-up" style="animation-delay: ${index * 0.1}s">
            <div class="result-header">
                <span class="result-rank">
                    <i class="fas fa-hashtag"></i> ${result.rank}
                </span>
                <span class="result-score">${result.score.toFixed(4)}</span>
            </div>
            <div class="result-content">${escapeHtml(result.content.substring(0, 300))}${result.content.length > 300 ? '...' : ''}</div>
            <div class="result-source">${result.source}</div>
        </div>
    `).join('');
}

// Update Chart
function updateChart(result) {
    const labels = result.raptor.results.map((_, i) => `Rank ${i + 1}`);
    const raptorScores = result.raptor.results.map(r => r.score);
    const graphragScores = result.graphrag.results.map(r => r.score);
    
    comparisonChart.data.labels = labels;
    comparisonChart.data.datasets[0].data = raptorScores;
    comparisonChart.data.datasets[1].data = graphragScores;
    comparisonChart.update();
}

// Load Sample Document
async function loadSampleDocument() {
    showLoading(true);
    
    try {
        const result = await apiCall('/api/sample-document');
        elements.documentInput.value = result.text;
        showToast('Sample document loaded!', 'info');
    } catch (error) {
        showToast(`Error loading sample: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Toggle Sample Queries
async function toggleSampleQueries() {
    const dropdown = elements.sampleQueriesDropdown;
    
    if (dropdown.classList.contains('active')) {
        dropdown.classList.remove('active');
        return;
    }
    
    try {
        const result = await apiCall('/api/sample-queries');
        
        elements.sampleQueriesList.innerHTML = result.queries.map(query => `
            <li onclick="selectQuery('${escapeHtml(query)}')">${escapeHtml(query)}</li>
        `).join('');
        
        dropdown.classList.add('active');
    } catch (error) {
        showToast(`Error loading queries: ${error.message}`, 'error');
    }
}

// Select Query
function selectQuery(query) {
    elements.queryInput.value = query;
    elements.sampleQueriesDropdown.classList.remove('active');
}

// Handle File Upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const ext = file.name.split('.').pop().toLowerCase();
    const binaryTypes = ['pdf', 'doc', 'docx'];
    
    if (binaryTypes.includes(ext)) {
        // For PDF/DOC files, store for direct upload
        pendingFile = file;
        elements.documentInput.value = `[File ready to upload: ${file.name}]\n\nClick "Index Document" to process this ${ext.toUpperCase()} file.`;
        elements.documentInput.readOnly = true;
        showToast(`${ext.toUpperCase()} file selected: ${file.name}. Click "Index Document" to process.`, 'info');
    } else {
        // For text files, read content
        pendingFile = null;
        elements.documentInput.readOnly = false;
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.documentInput.value = e.target.result;
            showToast(`File loaded: ${file.name}`, 'info');
        };
        reader.onerror = () => {
            showToast('Error reading file', 'error');
        };
        reader.readAsText(file);
    }
}

// Load Stats
async function loadStats() {
    try {
        const result = await apiCall('/api/stats');
        
        elements.raptorWins.textContent = result.query_history.raptor_wins;
        elements.graphragWins.textContent = result.query_history.graphrag_wins;
        
        updateHistory(result.recent_queries);
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Update History
function updateHistory(history) {
    if (!history || history.length === 0) {
        elements.historyList.innerHTML = '<p class="no-history">No queries yet. Start comparing!</p>';
        return;
    }
    
    elements.historyList.innerHTML = history.reverse().map(item => `
        <div class="history-item" onclick="selectQuery('${escapeHtml(item.query)}')">
            <span class="history-query">${escapeHtml(item.query)}</span>
            <span class="history-winner ${item.winner.toLowerCase()}">${item.winner}</span>
            <span class="history-time">${formatTime(item.timestamp)}</span>
        </div>
    `).join('');
}

// Clear History
async function clearHistory() {
    if (!confirm('Are you sure you want to clear all data?')) return;
    
    showLoading(true);
    
    try {
        await apiCall('/api/clear', { method: 'POST' });
        
        isDocumentIndexed = false;
        pendingFile = null;
        elements.documentInput.readOnly = false;
        updateIndexStatus('not-indexed');
        elements.resultsSection.style.display = 'none';
        elements.documentInput.value = '';
        elements.queryInput.value = '';
        loadStats();
        
        showToast('Data cleared successfully', 'success');
        
    } catch (error) {
        showToast(`Error clearing data: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Update Index Status
function updateIndexStatus(status, details = null) {
    const statusEl = elements.indexStatus.querySelector('.status-indicator');
    statusEl.className = `status-indicator ${status}`;
    
    switch (status) {
        case 'indexing':
            statusEl.innerHTML = '<i class="fas fa-circle"></i> Indexing document...';
            break;
        case 'indexed':
            let text = 'Document indexed';
            if (details) {
                text += ` (RAPTOR: ${details.raptor.chunks} chunks, GraphRAG: ${details.graphrag.entities} entities)`;
            }
            statusEl.innerHTML = `<i class="fas fa-check-circle"></i> ${text}`;
            break;
        default:
            statusEl.innerHTML = '<i class="fas fa-circle"></i> No document indexed';
    }
}

// Show Loading
function showLoading(show) {
    elements.loadingOverlay.classList.toggle('active', show);
}

// Show Toast
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas fa-${icons[type]}"></i>
        <span>${message}</span>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

// Make selectQuery available globally
window.selectQuery = selectQuery;
