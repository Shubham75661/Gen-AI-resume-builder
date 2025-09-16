import re
import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from dotenv import load_dotenv

# ------------------- CONFIG -------------------
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# ------------------- MODEL -------------------
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# ------------------- STATIC DATA -------------------
FULL_SKILLS = {
    "Programming Languages": ["JavaScript", "Python", "Java", "C++", "TypeScript", "SQL"],
    "AI/ML": ["scikit-learn", "NumPy", "pandas", "matplotlib", "Streamlit", "regression modeling"],
    "Frontend Frameworks": ["React.js", "Electron.js", "Next.js", "HTML", "CSS"],
    "Backend Frameworks": ["Node.js", "Flask", "Django"],
    "Databases": ["PostgreSQL", "MongoDB", "DynamoDB", "Redis"],
    "DevOps Tools": ["Docker", "Kubernetes", "AWS", "Postman", "Jenkins", "Git", "GitHub"]
}

FULL_EXPERIENCE = """
Add you're experience
"""

# ------------------- PROMPT TEMPLATE -------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that tailors resumes based on job descriptions.\n\n"
     "Here is the candidate's complete skills and experience:\n"
     "SKILLS:\n{skills}\n\n"
     "EXPERIENCE:\n{experience}\n\n"
     "Instructions:\n"
     "1. Analyze the job description and select the most relevant experiences for each role\n"
     "2. For each job role, provide EXACTLY 3 bullet points that are most relevant to the job description\n"
     "3. Reorder and modify bullet points to match job requirements\n"
     "4. Wrap important technologies, tools, frameworks, and metrics in [[B]] and [[/B]] tags\n"
     "5. Include all job roles but focus on the most relevant experiences\n"
     "6. Do NOT use markdown formatting\n\n"
     "Output format:\n"
     "SKILLS:\n"
     "Category: skill1, skill2, skill3\n"
     "Category: skill1, skill2\n\n"
     "EXPERIENCE:\n"
     "Role, Company — Dates, Location\n"
     "• bullet point with [[B]]important tech[[/B]]\n"
     "• another bullet point\n"
     ),
    ("user", "Job Description: {job_description}")
])

parser = StrOutputParser()
chain = prompt | model | parser

# ------------------- UTILITY FUNCTIONS -------------------

def parse_output(output_text: str):
    """Parse the AI output into skills and experience sections"""
    if "EXPERIENCE:" in output_text:
        parts = output_text.split("EXPERIENCE:")
        skills_text = parts[0].replace("SKILLS:", "").strip()
        experience_text = parts[1].strip()
    else:
        skills_text, experience_text = output_text.strip(), ""
    return skills_text, experience_text

def parse_skills_from_text(skills_text: str) -> dict:
    """Parse skills text into a structured dictionary"""
    skills_dict = {}
    lines = [line.strip() for line in skills_text.split('\n') if line.strip()]
    
    for line in lines:
        if ':' in line:
            category, skills_str = line.split(':', 1)
            category = category.strip()
            skills_list = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
            if skills_list:  # Only add if there are skills
                skills_dict[category] = skills_list
    
    return skills_dict

def clean_and_format_experience(exp_text: str) -> str:
    """Clean and format experience text to match original format"""
    exp_text = exp_text.replace("EXPERIENCE:", "").replace("EXPERIENCE", "").strip()
    lines = exp_text.splitlines()
    formatted = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Handle bullet points
        if line.startswith("•"):
            formatted.append(line)
        else:
            # This is a role line - ensure proper spacing
            if formatted and formatted[-1] != "":
                formatted.append("")  # add spacing before new role
            formatted.append(line)
    
    return "\n".join(formatted)

def format_bold_identifiers(text: str) -> str:
    """Convert bold markers to HTML tags"""
    text = re.sub(r'\[\[B\]\](.*?)\[\[/B\]\]', r'<b>\1</b>', text)
    text = re.sub(r'\[\[(.*?)\]\]', r'<b>\1</b>', text)
    return text

def format_skills_for_table(skills_dict):
    """Format skills dictionary for PDF table"""
    return [[category, ", ".join(skills)] for category, skills in skills_dict.items()]

# ------------------- PDF CREATION -------------------
def create_resume_pdf(skills_dict, experience_text, filename="custom_resume.pdf"):
    """Create a professional resume PDF"""
    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    content = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=6,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=12,
        spaceAfter=6,
        borderWidth=1,
        borderColor=colors.HexColor('#e2e8f0'),
        borderPadding=3,
        backColor=colors.HexColor('#f7fafc')
    )

    centered_style = ParagraphStyle(
        name="CenterBlock",
        parent=styles["Normal"],
        alignment=TA_CENTER
    )
    # Header
    content.append(Paragraph("SHUBHAM KOTARWAR", title_style))
    content.append(Paragraph(
        "+91 123456789 ■ Pune, MH<br/>",
         centered_style
    ))
    content.append(Spacer(10, 12))

    content.append(Paragraph(
        "examplemail@gmail.com ■ linkedin.com/in/username ■ leetcode/username",
        centered_style
    ))
    content.append(Spacer(1, 12))

    # Summary
    content.append(Paragraph("SUMMARY", heading_style))
    content.append(Paragraph(
        "Summary",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    # Education
    content.append(Paragraph("EDUCATION", heading_style))
    content.append(Paragraph(
        "Education",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    # Skills table
    if skills_dict:
        content.append(Paragraph("SKILLS", heading_style))
        table_data = format_skills_for_table(skills_dict)
        table = Table(table_data, colWidths=[125, 295])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),  # Bold category names
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        content.append(table)
        content.append(Spacer(1, 12))

    # Experience with bold markers parsed
    content.append(Paragraph("EXPERIENCE", heading_style))
    if experience_text:
        formatted_exp = clean_and_format_experience(experience_text)
        for line in formatted_exp.split("\n"):
            if not line.strip():
                content.append(Spacer(1, 6))
            else:
                bold_line = format_bold_identifiers(line)
                if line.strip().startswith('•'):
                    # Bullet points
                    content.append(Paragraph(bold_line, styles["Normal"]))
                else:
                    # Role titles
                    role_style = ParagraphStyle(
                        'RoleStyle',
                        parent=styles['Normal'],
                        fontName='Helvetica-Bold',
                        fontSize=11,
                        spaceBefore=6
                    )
                    content.append(Paragraph(bold_line, role_style))

    # Projects
    content.append(Spacer(1, 12))
    content.append(Paragraph("PROJECTS", heading_style))
    content.append(Paragraph(
        "Projects",
        styles["Normal"]
    ))

    # Achievements
    content.append(Spacer(1, 12))
    content.append(Paragraph("ACHIEVEMENTS", heading_style))
    content.append(Paragraph(
        "ACHIEVEMENTS",
        styles["Normal"]
    ))

    doc.build(content)
    print(f"Resume generated: {filename}")
    return filename

# ------------------- FASTAPI APP -------------------
app = FastAPI(title="AI Resume Generator", version="1.0", description="Tailored Resume Generator using Groq AI")

@app.get("/")
async def root():
    return {"message": "AI Resume Generator API", "endpoints": ["/generate_pdf", "/download_resume"]}

@app.post("/generate_pdf")
async def generate_pdf(request: Request):
    """Generate a tailored resume PDF based on job description"""
    try:
        jd = (await request.body()).decode("utf-8").strip()
        if not jd:
            return JSONResponse(content={"error": "Job description required"}, status_code=400)

        # Generate tailored content using AI
        output_text = chain.invoke({
            "skills": json.dumps(FULL_SKILLS, indent=2),
            "experience": FULL_EXPERIENCE,
            "job_description": jd
        })
        
        # Parse the output
        skills_text, experience_text = parse_output(output_text)
        skills_dict = parse_skills_from_text(skills_text)
        
        # Create PDF
        filename = create_resume_pdf(skills_dict, experience_text, "custom_resume.pdf")
        
        return {
            "message": "Resume generated successfully",
            "filename": filename,
            "skills_selected": list(skills_dict.keys()),
            "download_url": "/download_resume"
        }
        
    except Exception as e:
        return JSONResponse(content={"error": f"Error generating resume: {str(e)}"}, status_code=500)

@app.get("/download_resume")
async def download_resume():
    """Download the generated resume PDF"""
    filename = "custom_resume.pdf"
    if os.path.exists(filename):
        return FileResponse(
            filename,
            media_type="application/pdf",
            filename="Shubham_Kotarwar_Resume.pdf"
        )
    else:
        return JSONResponse(content={"error": "Resume not found. Please generate one first."}, status_code=404)

@app.post("/preview")
async def preview_resume(request: Request):
    """Preview the tailored skills and experience without generating PDF"""
    try:
        jd = (await request.body()).decode("utf-8").strip()
        if not jd:
            return JSONResponse(content={"error": "Job description required"}, status_code=400)

        output_text = chain.invoke({
            "skills": json.dumps(FULL_SKILLS, indent=2),
            "experience": FULL_EXPERIENCE,
            "job_description": jd
        })
        
        skills_text, experience_text = parse_output(output_text)
        selected_skills_dict = parse_skills_from_text(skills_text)
        final_skills_dict = merge_skills_with_selected(selected_skills_dict, FULL_SKILLS)
        
        return {
            "tailored_skills": final_skills_dict,
            "tailored_experience": experience_text,
            "raw_output": output_text
        }
        
    except Exception as e:
        return JSONResponse(content={"error": f"Error previewing resume: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="127.0.0.1", port=3001, reload=True)