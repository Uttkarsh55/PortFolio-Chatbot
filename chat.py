import os 
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel
import re

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError('API key not found or Empty.')

client=Groq(api_key=my_api_key)
model="llama-3.3-70b-versatile"

def ask_llm(messages):
    response=client.chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in response:
        content=chunk.choices[0].delta.content
        if content:
            print(content, end='', flush=True)


def read_profile(profile_data):
    whole_data=""
    with open(profile_data,'r') as f:
        content=f.readlines()
        for line in content:
            whole_data += line
    clean_data(whole_data)
    return whole_data

def clean_data(whole_data):
    clean_data=whole_data.strip('')
    return clean_data

def jd_parsing(query):
    system_prompt=f'''
    You are HR Assistant who evaluates the user on the basis of the Job description uploaded by the user.
    You can access the user skills from {schema} and extract the skills.
    Compare the skills the user have with the skills asked in the Job Description {query}.
    Do not invent any other skills if the user does not have it.
    Do not hallucinate.
    At last evaluate the User if he/she is a good fit for the Job Description Uploaded.
    Also provide a verdict and reason for why you came to that conclusion.
    Do not provide a lot of text.
    
    Just give the output as :
    User Skills : skills
    skills asked in JD : skills asked in JD
    matched skills : skills which are present in both jd and user profile
    
    final verdict : fit for the job or not
    Reason : what is the reason for rejecting or accepting the user for the Job.
    '''
    message=[{
        {
        'role':'system',
        'content':system_prompt
        },
        {
            'role':'user',
            'content':query
        }
        }]
    
    ask_llm(message)
    
    
    
    
class Profile(BaseModel):
    name:str
    age:int
    introduction:str
    skills:list[str]
    projects:list[str]
    experience:list[str]
    achievements:list[str]
    certificates:list[str]
    
schema=Profile.model_json_schema()


response_format={
    'type':'json_object'
}

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
profile_path=r"D:\AI\8 weeks ai\week2\MP2\profile_info\uttkarsh_monga_ai_engineer_profile_final.txt"
user_data=read_profile(profile_path)

while True:
    user_prompt=input("\nEnter what you wanna ask about me?: ")
    if user_prompt == 'n' or not user_prompt:
        break
    else:
        messages=[
            message,
            {
                'role':'user',
                'content':user_data
            },
            {
                'role':'user',
                'content':user_prompt
            }
        ]
        answer=ask_llm(messages)
        
        if answer:
            query=user_prompt
            if 'jd' in query or "job description" in query:
                jd_parsing(query)
            
            messages=messages.append(
                {
                'role':'user',
                'content':user_prompt
                },
                {
                    'role':'assistant',
                    'content':answer
                }
                                     
            )
        