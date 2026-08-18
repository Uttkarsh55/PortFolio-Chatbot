import os
import json
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
# from doc import Document
from pydantic import BaseModel
from pathlib import Path
import json
load_dotenv()

BASE_DIR=Path(__file__).resolve().parent
profile_path= BASE_DIR / "profile_info" / "uttkarsh_monga_ai_engineer_profile_final.txt"

# print(BASE_DIR)
# print(profile_path)


my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError('API key not found or Empty.')

model='llama-3.3-70b-versatile'
client=Groq(api_key=my_api_key)


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


def ask_llm(question):
        # messages=[message]
    # messages=question
    # if 'jd' or 'Job description'.lower() in question.lower():
    #     jd_json=read_jd(question,jd_schema)
    #     comparison=comparison(jd_json)
    #     print(comparison)
    
    whole_response=""
    response=client.chat.completions.create(model=model,messages=question,stream=True)
    for chunk in response:
        content=chunk.choices[0].delta.content
        if content:
            print(content,end="",flush=True)
            whole_response+=content + ' '
    
    return whole_response   

def system_call_llm(system_prompt):
    message = {
        'role':'system',
        'content':system_prompt
    }
    
    response=client.chat.completions.create(model=model, messages=[message], response_format=response_format)
        
    return response.choices[0].message.content


def read_profile():
    with open('user_profile_data.json','r') as file:
        user_data=file.read()    
        # user_data=json.loads(user_data)
        return user_data
    # whole_data=""
    # with open(path,'r') as file:
    #     content=file.readlines()
    #     for line in content:
    #         whole_data += line
    
    # system_prompt=f'''
    # You are an HR assistant who is Expert at reading Resume.
    # Your task is to Read the candidate profile data {whole_data} and provide the output in JSON format Strictly {profile_schema}.
    # Do Not Hallucinate and do not invent any Information by Yourself.
    # '''
    # return system_call_llm(system_prompt)    

response_format={
    'type':'json_object'
}

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

# jd =""

system_prompt=f'''
You are the AI Representative of this person whom information I am giving You.
You will only answer the question which are asked from you without inventing new information and only answering from the data given to you.
be Honest.
If any information is missing just answer with I don't know about that.
never hallucinate.
'''

message={
    'role':'system',
    'content':system_prompt
}

user_data=read_profile()
messages=[
            message,
            {
                'role':'user',
                'content':user_data
            }
        ]
while True:
    question=input("\nuser: ")
    if not question or question.lower() in ['n','q','exit','quit']:
        break
    else:
        if is_jd(question):
            jd_json=read_jd(question,jd_schema)
            jd_verdict=comparison(jd_json,user_data)
            print(jd_verdict)
            messages.extend([
                                {
                                    'role':'user',
                                    'content':question
                                },
                                {
                                    'role':'assistant',
                                    'content':jd_verdict
                                }
                           ])
        else:
            messages.append({
                'role':'user',
                'content':question
            })
            answer=ask_llm(messages)
            if answer:
                messages.append(
                                {
                                    'role':'assistant',
                                    'content':answer
                                }
            )