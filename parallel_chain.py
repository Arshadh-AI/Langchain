import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
#LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")

#os.environ["LANGCHAIN_TRACING_V2"]=LANGCHAIN_TRACING_V2
os.environ["LANGSMITH_API_KEY"]=LANGSMITH_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
# Create a ChatOpenAI model
#model = ChatOpenAI(model="gpt-4o")
#model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

model = ChatOpenAI(
    model_name="deepseek/deepseek-r1-distill-llama-8b",
    temperature=0.7
)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert movie reviewer."),
        ("human", "give the story of the movie {Movie_Name}."),
    ]
)


# Define pros analysis step
def analyze_pros(story):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert movie reviewer."),
            (
                "human",
                "Given this story: {story}, list the pros of this story.",
            ),
        ]
    )
    return pros_template.format_prompt(story=story)


# Define cons analysis step
def analyze_cons(story):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert movie reviewer."),
            (
                "human",
                "Given this story: {story}, list the cons of this story.",
            ),
        ]
    )
    return cons_template.format_prompt(story=story)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"Movie_Name": "Sholay"})

# Output
print(result)
