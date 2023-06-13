# Dependencies
import os
from apikey import apikey

import streamlit as st # application framework
from langchain.llms import OpenAI # Give AI service to leverage large language model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# makes apikey available to openai service by setting a dictionary key
os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('Email Genuis') 
prompt = st.text_input('Enter the job industry you are actively pursuing:')

# Prompt templates
companies_template = PromptTemplate(
    input_variables = ['industry'],
    template = 'What are the top {industry} companies to apply to for a summer internship as an undergraduate student?'
)

email_template = PromptTemplate(
    input_variables = ['companies'],
    template = 'Write me a detailed email addressed to {companies} requesting an internship as a college student.'
)

memory = ConversationBufferMemory(input_key = 'industry', memory_key = 'chat_history')

# dictate creativity of large language model
llm = OpenAI(temperature = 0.5)
companies_chain = LLMChain(llm =llm, prompt = companies_template, verbose = True, output_key = 'companies', memory = memory)
email_chain = LLMChain(llm =llm, prompt = email_template, verbose = True, output_key = 'email', memory = memory)
# runs chains in sequential order
sequential_chain = SequentialChain(chains = [companies_chain, email_chain], input_variables = ['industry'], output_variables = ['companies', 'email'], verbose = True)

# Show response if prompt entered
if prompt:
    response = sequential_chain({'industry': prompt})
    st.write("Potential companies within your desired industry:" + response['companies'])
    st.write("Email template for recruiters at the companies listed above: " + response['email'])

    with st.expander('Message History'):
        st.info(memory.buffer)
