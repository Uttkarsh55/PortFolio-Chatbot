import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

my_api_key=os.getenv('GROQ_API_KEY')
if not my_api_key:
    raise ValueError(f"API key empty {my_api_key}")

client=Groq(api_key=my_api_key)
model='openai/gpt-oss-20b'

app=FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

#load profile once at startup

with open('user_profile_data.json','r') as f:
    USER_PROFILE=f.read()
    

class Message(BaseModel):
    role:str
    content:str
    
class ChatRequest(BaseModel):
    messages:list[Message]

class Experience(BaseModel):
    company_name : str | None = None
    duration : int | None = None
    skills_used : list[str] | None = None
    role : str | None = None
    description : str | None = None
    
# experience_schema=Experience.model_json_schema()
    
class Profile(BaseModel):
    name : str
    age : str
    email : str
    profile : str
    mobile_number : int
    college : str
    education : list[str]
    total_experience_years : float | None = None
    experience : list[Experience] = [] 
    skills : list[str]
    projects : list[str] | None
    certifications : list[str]
    achievements : list[str]
    
profile_schema=Profile.model_json_schema()
    
class Job_Description(BaseModel):
    role:str
    description:str
    required_experience : str
    skills : list[str]
    required_education : str
    location : str

jd_schema=Job_Description.model_json_schema()
    
response_format={
    'type':'json_object'
}

def ask_llm(question):
        # messages=[message]
    # messages=question
    # if 'jd' or 'Job description'.lower() in question.lower():
    #     jd_json=read_jd(question,jd_schema)
    #     comparison=comparison(jd_json)
    #     print(comparison)
    response=client.chat.completions.create(model=model,messages=question)
    answer=response.choices[0].message.content
    return answer   


def read_jd(question,jd_schema):
    system_prompt=f'''
    You are an HR assistant who is an Expert at reading Job Description.
    Your job is to read the Job description from the input given by user {question} and give the output in a structured JSON format as per the schema {jd_schema}.
    Do not invent any information by yourself and do not Hallucinate.
    '''
    message=[{
        'role':'system',
        'content':system_prompt
                    },
             {
                'role':'user',
                'content':question 
             }
             ]
    response=client.chat.completions.create(model=model, messages=message, response_format=response_format)
    
    return response.choices[0].message.content


def comparison(jd,user_profile):
    # user_profile=read_profile()
    system_prompt=f'''
    You are an HR and you are an expert at evaluating the resume and comparing it with the Job description {jd}.
    Your task is to compare the Job description provided by the User in the question with the Candidate profile data {user_profile} which is the data extracted from the resume of the candidate.
    Compare them honestly and do not favour the candidate.
    Do not hallucinate.
    Do not invent any other information about the candidate by yourself.
    Provide the Ouput by giving a Match Score out of 100 and giving reason why the candidate is good fit for the role or not.
    Strictly follow the Given output format below and do not provide any other type of information.
    the Ouput should be in following format : 
    Match Score : Score percentage
    Answer : Candidate pass or Fail
    Reason : reason for passing or failing the candidate.
    '''
    messages=[{
        'role':'system',
        'content':system_prompt
    }]
    response=client.chat.completions.create(model=model,messages=messages)
    return response.choices[0].message.content

def is_jd(question):
    if len(question)<200:
        return False
    prompt=f'''
    Analyze the following text and determine if it is a Job Description/Posting.
    Reply with ONLY the word "YES" or "NO".
    
    Text: {question[:1000]} # Only need to check the first 1000 chars
    '''
    response=client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{
            'role':'user',
            'content':prompt
        }],
        temperature=0)
    return 'YES' in response.choices[0].message.content


@app.post('/chat')
async def chat_endpoint(request: ChatRequest):
    # 1. Define the strict persona and inject your resume data
    system_prompt = f"""
    You are the AI Representative of Uttkarsh Monga.
    You will only answer questions based on the data given to you.
    Be honest. If any information is missing, just answer with "I don't know about that".
    Never hallucinate or talk about being an AI language model.

    Here is Uttkarsh's Profile Data:
    {USER_PROFILE}
    """

    # 2. Start the Groq message array with the System Prompt
    messages_for_groq = [{"role": "system", "content": system_prompt}]
    
    # 3. Append the user's chat history from the frontend
    messages_for_groq.extend([msg.model_dump() for msg in request.messages])
    
    # 4. Call Groq
    answer = ask_llm(messages_for_groq)
    return {"response": answer}    
@app.post('/evaluate-jd')
async def evaluate_jd_endpoint(file:UploadFile=File(...)):
    # Read the file directory into memory
    content=await file.read()
    jd_text=content.decode("utf-8")
    
    jd_json=read_jd(jd_text,jd_schema)
    verdict=comparison(jd_json, USER_PROFILE)
    
    return {
        "Filename":file.filename,
        "evaluation":verdict
    }