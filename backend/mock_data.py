customers = {
    '101': {
        'name': 'Rajesh Kumar Sharma',
        'loanType': 'Personal Loan',
        'outstanding': 'Rs 4,85,000',
        'emi': 'Rs 18,500',
        'age': '28 months',
        'branch': 'Mumbai Central',
        'riskScore': 73,
        'probability': 0.685,
        'confidence': 91.2,
        'riskLevel': 'High Risk',
        'salary_delay': 6,
        'utilization': 0.82,
        'drivers': [
            {'name': 'Income Drop', 'value': -35, 'percent': 85, 'color': '#ef4444'},
            {'name': 'High Credit Utilization', 'value': 82, 'percent': 82, 'color': '#f97316'}
        ],
        'timeline': [
            {'title': 'EMI Payment Missed', 'desc': 'February EMI bounced', 'date': 'Feb 05, 2026', 'type': 'critical'},
            {'title': 'Salary Deposit Delayed', 'desc': 'Salary late', 'date': 'Jan 20, 2026', 'type': 'warning'}
        ]
    },
    '102': {
        'name': 'Priya Desai',
        'loanType': 'Home Loan',
        'outstanding': 'Rs 28,50,000',
        'emi': 'Rs 45,200',
        'age': '42 months',
        'branch': 'Bangalore Tech Park',
        'riskScore': 82,
        'probability': 0.792,
        'confidence': 94.5,
        'riskLevel': 'Critical',
        'salary_delay': 2,
        'utilization': 0.45,
        'drivers': [
            {'name': 'Multiple EMI Bounces', 'value': '3 in 4 months', 'percent': 92, 'color': '#dc2626'}
        ],
        'timeline': [
            {'title': 'Third EMI Bounce', 'desc': 'Payment failed', 'date': 'Feb 10, 2026', 'type': 'critical'}
        ]
    },
    '103': {
        'name': 'Amit Patel',
        'loanType': 'Auto Loan',
        'outstanding': 'Rs 3,25,000',
        'emi': 'Rs 12,800',
        'age': '18 months',
        'branch': 'Delhi NCR',
        'riskScore': 71,
        'probability': 0.658,
        'confidence': 88.3,
        'riskLevel': 'High Risk',
        'salary_delay': 1,
        'utilization': 0.78,
        'drivers': [
            {'name': 'Late Payments', 'value': '4 in 6 months', 'percent': 75, 'color': '#ef4444'}
        ],
        'timeline': [
            {'title': 'Payment Delayed', 'desc': 'EMI paid late', 'date': 'Feb 08, 2026', 'type': 'warning'}
        ]
    }
}

dashboard_data = {
    'safetyScore': '84.2',
    'safetyTrend': '+3.2%',
    'activeProtected': '24,582',
    'accountsTrend': '+5.8%',
    'successRate': '76.3%',
    'successTrend': '+4.1%',
    'portfolio_type': {
        'labels': ['Personal Loan', 'Home Loan', 'Auto Loan', 'Business Loan', 'Education Loan'],
        'data': [285.4, 398.7, 95.6, 52.3, 13.2]
    },
    'tenure': {
        'labels': ['0-1 Year', '1-3 Years', '3-5 Years', '5-10 Years', '10+ Years'],
        'data': [3245, 8967, 6821, 4234, 1315]
    },
    'portfolio_growth': {
        'labels': ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'],
        'data': [724, 752, 781, 802, 825, 845]
    },
    'disbursement': {
        'labels': ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'],
        'disbursements': [95, 108, 112, 98, 115, 122],
        'collections': [88, 102, 107, 94, 110, 118]
    },
    'recentAlerts': [
        {'segment': 'Personal Loan (Tier 2)', 'score': '62.4', 'level': 'High', 'status': 'Immediate Review'},
        {'segment': 'SME Retail', 'score': '71.2', 'level': 'Medium', 'status': 'Under Observation'},
        {'segment': 'Auto Loan (Regional)', 'score': '58.9', 'level': 'High', 'status': 'Manual Intervention'}
    ]
}

portfolio_data = {
    'totalValue': '₹845.2Cr',
    'valueTrend': '+12.4%',
    'healthyCount': '22,735',
    'healthyPercent': '92.5%',
    'avgScore': '78.5',
    'productPerformance': [
        {'type': 'Personal Loan', 'active': '9,847', 'value': '₹285.4 Cr', 'ticket': '₹2.9 L', 'score': '71.2', 'risk': 'Medium', 'status': 'Monitor'},
        {'type': 'Home Loan', 'active': '6,234', 'value': '₹398.7 Cr', 'ticket': '₹6.4 L', 'score': '84.1', 'risk': 'Low', 'status': 'Healthy'},
        {'type': 'Auto Loan', 'active': '4,892', 'value': '₹95.6 Cr', 'ticket': '₹1.9 L', 'score': '79.5', 'risk': 'Low', 'status': 'Healthy'},
        {'type': 'Business Loan', 'active': '2,456', 'value': '₹52.3 Cr', 'ticket': '₹2.1 L', 'score': '62.8', 'risk': 'High', 'status': 'Watch'},
        {'type': 'Education Loan', 'active': '1,153', 'value': '₹13.2 Cr', 'ticket': '₹1.1 L', 'score': '88.4', 'risk': 'Low', 'status': 'Excellent'}
    ]
}

operations_data = [
    {'id': '101', 'name': 'Rajesh Kumar Sharma', 'product': 'Personal Loan', 'score': 73, 'risk': 'Medium', 'action': 'Assign RM + Restructure', 'status': 'In Progress', 'statusRisk': 'Medium'},
    {'id': '102', 'name': 'Priya Desai', 'product': 'Home Loan', 'score': 82, 'risk': 'Critical', 'action': 'Immediate Contact', 'status': 'Pending', 'statusRisk': 'High'},
    {'id': '103', 'name': 'Amit Patel', 'product': 'Auto Loan', 'score': 71, 'risk': 'High', 'action': 'Send Reminder', 'status': 'Completed', 'statusRisk': 'Low'}
]

model_data = {
    'accuracy': '89.4%',
    'f1': '0.865',
    'lastRetrained': 'Feb 1, 2026'
}
