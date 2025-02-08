import os
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables from .env
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
#LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")

#os.environ["LANGCHAIN_TRACING_V2"]=LANGCHAIN_TRACING_V2
os.environ["LANGSMITH_API_KEY"]=LANGSMITH_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
# Create a ChatOpenAI model
#model = ChatOpenAI(model="gpt-4o")
model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)