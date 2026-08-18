import os
from pathlib import Path
from groq import Groq
from pydantic import BaseModel
import json
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd
from docx import Document

BASE_DIR=Path(__file__).resolve().parent


load_dotenv()
my_api_key=os.getenv('GROQ_API_KEY')
client=Groq(api_key=my_api_key)

model='openai/gpt-oss-120b'



def get_resume_file(folder):
    folder = folder / "Resume"
    if not folder.exists():
        print("Folder was not present.")
        os.mkdir('Resume')
        print(f"Created folder {folder}. Please Put your resume in it.")
        return None
    
    for file in folder.iterdir():
        if file.is_file() and  file.suffix.lower() in ['.pdf','.docx','.doc','.txt']:
            return file
    

def extract_text(resume):
    
    if resume is None:
        print( "Resume was not added to the Resume Folder. Please add the Resume in it.")
        return ''
    try:
        text=""
        suffix=resume.suffix.lower()
        if suffix=='.pdf':
            reader=PdfReader(resume)
            for page in reader.pages:
                extracted=page.extract_text()
                if extracted:
                    text+=extracted+ '\n'
                
            return text
        
        elif suffix in ['.docx','.doc']:
                doc=Document(resume)
                for para in doc.paragraphs:
                    text+=para.text + '\n'
                    
                return text
            
        elif suffix == '.txt':
            with open(resume,'r') as file:
                content=file.readlines()
                text=""
                for line in content:
                    text+=line
            return text
    except Exception as e:
        print(f"Error Reading the file {resume.name}: {e}")
        return '' 
        

class Experience(BaseModel):
    company_name : str | None = None
    duration : int | None = None
    skills_used : list[str] | None = None
    role : str | None = None
    description : str | None = None
    
class Profile(BaseModel):
    name : str
    age : str
    email : str
    profile : str
    mobile_number : int
    college : str
    education : list[str]
    total_experience_years : str | None = None
    experience : list[Experience] = [] 
    skills : list[str]
    projects : list[str] | None
    certifications : list[str]
    achievements : list[str]
    
profile_schema=Profile.model_json_schema()
response_format={
    'type':'json_object'
}
system_prompt=f'''
     You are an expert resume parser.
    
        Extract information from the resume based on its meaning,
        not only based on exact section headings.
    
        Different resumes may use different headings.
    
        For example:
        - Experience
        - Professional Experience
        - Work History
        - Employment
        - Internships
    
        These may all contain relevant experience.
        Do Not Extract Experience From heading Named "PROJECTS" or "projects".
        If the Duration of the Internship is not present, Calculate it by the gap between the start and end month of the Internship and add
        Skills may also appear in the skills section, work experience,
        internships or projects.
    
        Return ONLY valid JSON matching this schema:
    
        {profile_schema}
    
        Important rules:
    
        1. Do not invent information.
        2. If a value is not available, return null.
        3. If a list has no information, return an empty list.
        4. Include internships inside experiences.
        5. Extract skills mentioned across the entire resume.
        """
    '''
def text_to_json(data,system_prompt):
    message=[
        {
            'role':'system',
            'content':system_prompt
        },
        {
            'role':'user',
            'content':data
        }
    ]
    response=client.chat.completions.create(model=model, messages=message,response_format=response_format)
    return response.choices[0].message.content

resume=get_resume_file(BASE_DIR)

if resume == '':
    print("Resume Not found in Folder")
else:
    data=extract_text(resume)
    llm_ans=text_to_json(data,system_prompt)
    json_object=json.loads(llm_ans)
    with open('user_profile_data.json','w') as file:
        json.dump(json_object, file,  indent=4)
        print('JSON file created Successfully.')




