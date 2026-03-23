const COLUMNS = [
    "age_at_marriage",
    "marriage_duration_years",
    "num_children",
    "education_level",
    "employment_status",
    "combined_income",
    "religious_compatibility",
    "cultural_background_match",
    "communication_score",
    "conflict_frequency",
    "conflict_resolution_style",
    "financial_stress_level",
    "mental_health_issues",
    "infidelity_occurred",
    "counseling_attended",
    "social_support", "shared_hobbies_count",
    "marriage_type",
    "pre_marital_cohabitation",
    "domestic_violence_history",
    "trust_score"
];

const TRANSLATE = {
    education_level: {
        'No Formal Education': 'Không có bằng cấp',
        'High School': 'Trung học phổ thông',
        'Bachelor': 'Cử nhân',
        'Master': 'Thạc sĩ',
        'PhD': 'Tiến sĩ'
    },
    employment_status: {
        'Full-time': 'Toàn thời gian',
        'Part-time': 'Bán thời gian',
        'Unemployed': 'Thất nghiệp',
        'Homemaker': 'Nội trợ'
    },
    religious_compatibility: {
        'Same Religion': 'Cùng tôn giáo',
        'Different Religion': 'Khác tôn giáo',
        'Not Religious': 'Không theo tôn giáo'
    },
    conflict_resolution_style: {
        'Collaborative': 'Hợp tác',
        'Aggressive': 'Hung hăng',
        'Avoidant': 'Né tránh',
        'Passive': 'Thụ động'
    },
    marriage_type: {
        'Love': 'Tình yêu',
        'Arranged': 'Sắp đặt',
        'Other': 'Khác'
    }
};

const BINARY_FIELDS = [
    'cultural_background_match',
    'mental_health_issues',
    'infidelity_occurred',
    'counseling_attended',
    'pre_marital_cohabitation',
    'domestic_violence_history'
];

function translateValue(field, value) {
    if (TRANSLATE[field] && TRANSLATE[field][value] !== undefined) {
        return TRANSLATE[field][value];
    }
    if (BINARY_FIELDS.includes(field)) {
        return String(value) === '1' ? 'Có' : 'Không';
    }
    return value;
}


function loadHistory() {
    const history = JSON.parse(sessionStorage.getItem('predictionHistory') || '[]');
    const tbody = document.getElementById('historyBody');
    const emptyState = document.getElementById('emptyState');
    tbody.innerHTML = '';
    if (history.length === 0) {
        emptyState.style.display = 'block';
        return;
    }
    emptyState.style.display = 'none';
    history.forEach((entry, index) => {
        const tr = document.createElement('tr');
        const prediction = entry.prediction;
        const badgeClass = prediction === 1 ? 'badge-danger' : 'badge-success';
        const badgeText = prediction === 1 ? 'Ly hôn' : 'Không ly hôn';
        const d = entry.data;
        tr.innerHTML = `
            <td>${index + 1}</td>
            <td>${entry.timestamp || '—'}</td>
            <td>${d.age_at_marriage}</td>
            <td>${d.marriage_duration_years}</td>
            <td>${d.num_children}</td>
            <td>${translateValue('education_level', d.education_level)}</td>
            <td>${translateValue('employment_status', d.employment_status)}</td>
            <td>${d.combined_income}</td>
            <td>${translateValue('religious_compatibility', d.religious_compatibility)}</td>
            <td>${translateValue('cultural_background_match', d.cultural_background_match)}</td>
            <td>${d.communication_score}</td>
            <td>${d.conflict_frequency}</td>
            <td>${translateValue('conflict_resolution_style', d.conflict_resolution_style)}</td>
            <td>${d.financial_stress_level}</td>
            <td>${translateValue('mental_health_issues', d.mental_health_issues)}</td>
            <td>${translateValue('infidelity_occurred', d.infidelity_occurred)}</td>
            <td>${translateValue('counseling_attended', d.counseling_attended)}</td>
            <td>${d.social_support}</td>
            <td>${d.shared_hobbies_count}</td>
            <td>${translateValue('marriage_type', d.marriage_type)}</td>
            <td>${translateValue('pre_marital_cohabitation', d.pre_marital_cohabitation)}</td>
            <td>${translateValue('domestic_violence_history', d.domestic_violence_history)}</td>
            <td>${d.trust_score}</td>
            <td><span class="badge ${badgeClass}">${badgeText}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function savePrediction(data, prediction) {
    const history = JSON.parse(sessionStorage.getItem('predictionHistory') || '[]');
    const now = new Date();
    const timestamp = now.toLocaleString('vi-VN');
    history.push({
        timestamp: timestamp,
        data: data,
        prediction: prediction
    });

    sessionStorage.setItem('predictionHistory', JSON.stringify(history));
}

function showModal(prediction) {
    const modal = document.getElementById('resultModal');
    const icon = document.getElementById('resultIcon');
    const title = document.getElementById('resultTitle');
    const message = document.getElementById('resultMessage');
    if (prediction === 1) {
        icon.textContent = '💔';
        title.textContent = 'Dự đoán: Ly Hôn';
        title.style.color = 'var(--accent-secondary)';
        message.textContent = 'Dựa trên dữ liệu bạn cung cấp, mô hình dự đoán có khả năng ly hôn.';
    } else {
        icon.textContent = '💚';
        title.textContent = 'Dự đoán: Không Ly Hôn';
        title.style.color = 'var(--accent-success)';
        message.textContent = 'Dựa trên dữ liệu bạn cung cấp, mô hình dự đoán không có khả năng ly hôn.';
    }
    modal.classList.add('active');
}

function closeModal() {
    document.getElementById('resultModal').classList.remove('active');
}
document.getElementById('resultModal').addEventListener('click', function (e) {
    if (e.target === this) closeModal();
});

function clearHistory() {
    sessionStorage.removeItem('predictionHistory');
    loadHistory();
}
