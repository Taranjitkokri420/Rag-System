import os
from typing import Dict, Any, List
from pypdf import PdfReader

DEFAULT_CANDIDATES = [
    {"id": "mahesh_java", "file": "mahesh_java_developer.pdf", "name": "Mahesh", "role": "Sr. Full Stack Java Developer"},
    {"id": "sharath_java", "file": "sharath_java_developer.pdf", "name": "Sharath", "role": "Java Full Stack Developer"},
    {"id": "anudeep_java", "file": "anudeep_java_developer.pdf", "name": "Anudeep", "role": "Sr. Java Programmer"},
    {"id": "chetan_java", "file": "chetan_java_developer.pdf", "name": "Chetan Babu", "role": "Sr. Java Developer"},
    {"id": "pavan_java", "file": "pavan_java_developer.pdf", "name": "Pavan Kumar", "role": "Full Stack Java Developer"},
    {"id": "ashwini_java", "file": "ashwini_j2ee_developer.pdf", "name": "Ashwini", "role": "Sr. Java/J2EE Developer"},
    {"id": "sougandh_java", "file": "sougandh_java_developer.pdf", "name": "Sougandh", "role": "Java Fullstack Developer"},
    {"id": "nilesh_php", "file": "nilesh_php_lead.pdf", "name": "Nilesh Banthiya", "role": "PHP Lead / Web Developer"},
    {"id": "venkata_php", "file": "venkata_php_developer.pdf", "name": "Venkata", "role": "Sr. PHP / Drupal Developer"},
    {"id": "mani_hadoop", "file": "mani_hadoop_developer.pdf", "name": "Mani", "role": "Sr. Hadoop Developer"},
    {"id": "bapuji_hadoop", "file": "bapuji_hadoop_developer.pdf", "name": "Bapuji", "role": "Sr. Hadoop Developer"},
    {"id": "navneet_ba", "file": "navneet_business_analyst.pdf", "name": "Navneet", "role": "Business Analyst / Consultant"},
    {"id": "shail_ba", "file": "shail_business_analyst.pdf", "name": "Shail Tank", "role": "Business Analyst"},
    {"id": "krishna_bsa", "file": "krishna_business_systems_analyst.pdf", "name": "Krishna", "role": "Sr. Business Systems Analyst"},
    {"id": "amar_bsa", "file": "amar_business_systems_analyst.pdf", "name": "Amar", "role": "Business Systems Analyst"},
    {"id": "priyanka_bsa", "file": "priyanka_business_systems_analyst.pdf", "name": "Priyanka", "role": "Sr. Business Systems Analyst"},
    {"id": "avinash_pm", "file": "avinash_project_manager.pdf", "name": "Avinash", "role": "Project Manager"},
    {"id": "srivatsan_pm", "file": "srivatsan_project_manager.pdf", "name": "Srivatsan Ramabhadran", "role": "Project Manager / Business Data Analyst"},
    {"id": "adelina_pm", "file": "adelina_project_manager.pdf", "name": "Adelina Erimia", "role": "Project Manager (PMP)"},
    {"id": "adhi_scrum", "file": "adhi_scrum_master.pdf", "name": "Adhi Gopalam", "role": "Certified Scrum Master & Business Analyst"},
    {"id": "ranjan_pm_scrum", "file": "ranjan_pm_scrum_master.pdf", "name": "Ranjan Kumar Verma", "role": "Project Manager / Scrum Master"},
    {"id": "raju_tpm", "file": "raju_technical_program_manager.pdf", "name": "Raju Goduguchinta", "role": "Sr. Technical Program Manager"},
    {"id": "ravi_pm_devops", "file": "ravi_pm_devops.pdf", "name": "Ravi Prasad Burra", "role": "Certified PM / DevOps-Automation Architect"},
    {"id": "murali_pm_qa", "file": "murali_pm_qa.pdf", "name": "Muralidhar Chandrashekar", "role": "Project Manager (QA/Testing)"},
    {"id": "gautami_qa", "file": "gautami_qa_mobile_testing.pdf", "name": "Gautami Bulusu", "role": "QA / Mobile Test Engineer"},
]

def extract_pdf_text(path: str) -> str:
    """Extract all text contents from a PDF file."""
    if not os.path.exists(path):
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""

def load_resumes(resume_dir: str = "resumes") -> Dict[str, Dict[str, Any]]:
    """Load all resumes into memory mapped by candidate ID."""
    resumes = {}
    for c in DEFAULT_CANDIDATES:
        pdf_path = os.path.join(resume_dir, c["file"])
        text = extract_pdf_text(pdf_path)
        resumes[c["id"]] = {
            "name": c["name"],
            "role": c["role"],
            "file": c["file"],
            "text": text
        }
    return resumes
