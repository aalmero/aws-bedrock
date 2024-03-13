import json
from langchain_community.llms import Bedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Bedrock(
    credentials_profile_name="sre-sandbox-genai", model_id="anthropic.claude-v2"   
)

""" response = llm.invoke("how can langsmith help with testing?")

print(response) 
 """

# guide response with prompt template
prompt = ChatPromptTemplate.from_messages([("system", "You are a world class technical documentation writer."),
                                                  ("user","{input}")])

# simple output parser
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({"input": "how can langsmith help with testing?"})

print(response)



