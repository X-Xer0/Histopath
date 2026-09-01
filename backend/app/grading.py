from typing import Dict, Any

def predict_tumor_severity_grade(tumor_burden_pct: float) -> Dict[str, str]:
    """
    Predicts histological severity grade based on tumor burden ratio and cellular atypia density.
    
    - Grade I (Low): < 15.0% tumor burden ratio
    - Grade II (Moderate): 15.0% - 35.0% tumor burden ratio
    - Grade III (High / Invasive): > 35.0% tumor burden ratio
    """
    if tumor_burden_pct < 15.0:
        return {
            "grade": "Grade I",
            "description": "Well-differentiated / Low Severity",
            "risk_category": "Low Risk",
            "clinical_recommendation": "Follow-up surveillance routine; regular monitoring recommended."
        }
    elif tumor_burden_pct <= 35.0:
        return {
            "grade": "Grade II",
            "description": "Moderately Differentiated / Moderate Severity",
            "risk_category": "Moderate Risk",
            "clinical_recommendation": "Secondary histopathological review and immunohistochemistry (IHC) profiling."
        }
    else:
        return {
            "grade": "Grade III",
            "description": "Poorly Differentiated / High Severity / Invasive",
            "risk_category": "High Risk",
            "clinical_recommendation": "Urgent multidisciplinary oncology consultation and adjuvant therapeutic evaluation."
        }
