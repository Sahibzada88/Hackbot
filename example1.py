import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from info import pdf_database


@tool
def pdf_db(question):
    """
    This tool searches a database of PDFs for the best answer to a question.
    PDF is about hacking
    """
    return pdf_database(question)


tools = [pdf_db]

PROMPT = """You are a chatbot that speaks english.
You will answer every question which is related to the data in tools {tools}.
And if offtopic question is asked, the say "I am sorry".
"""

model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools)
messages = [SystemMessage(PROMPT)]

def chatbot(question):
    messages.append(HumanMessage(question))
    response = model.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "pdf_db":
                pdf_data = pdf_db(tool_call["args"]["question"])
                messages.append(ToolMessage(pdf_data,tool_call_id = tool_call["id"] ))
                
        response = model.invoke(messages)
        messages.append(response)
    return response.content

