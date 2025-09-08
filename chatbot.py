from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

#import the .env
from dotenv import load_dotenv
load_dotenv()

# config
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model = "")

# init the model 
llm = ChatOpenAI(temperature = 0.5, model = "")

# connect to the chromadb
vector_sctore = Chroma(
    collection_name = "cv_collections",
    embedding_function = embeddings_model,
    persist_directory = CHROMA_PATH,
)

# setup vectorstore to be retriever
num_results = 3
retriever = vector_sctore.as_retriever(search_kwargs={'k': num_results})

# call this function for every msg added to the chatbot
def stream_response(message, history):
    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)
    
    # add all chunks to 'knowledge'
    knowledge = ""
    
    for doc in docs:
        knowledge += doc.page_content+"\n\n"
        
    # make the call to the LLM (including prompt)
    if message is not None:
        
        partial_message =""
        
        rag_prompt = f""" 
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge.
        
        The question: {message}
        
        Conversation history: {history}
        
        The knowledge: {knowledge}
        
        """
        
        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
            
# init the gradio App
chatbot = gr.ChatInterface(stream_response, textbox = gr.Textbox(placeholder="Send to the LLM ...",
                                                                 container= False,
                                                                 autoscroll=True,
                                                                 scale=7
                                                                 ))
#launch the gradio App
chatbot.launch()