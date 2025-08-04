from langchain_google_genai import GoogleGenerativeAI # api for Google Gemini
from langchain_ollama.embeddings import OllamaEmbeddings # embeddings for model from Ollama 
from langchain_community.vectorstores import Chroma # chromadb is vector store for storing embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter # text splitter for splitting text into chunks
from langchain.prompts import PromptTemplate , ChatPromptTemplate# prompt template for generating queries
from langchain_core.output_parsers import StrOutputParser # output parser for parsing the output of the LLM as a string
from langchain_core.runnables import RunnablePassthrough # to pass the input through the chain without modification
from langchain.retrievers import MultiQueryRetriever # generates multiple queries from a single input query to retrieve relevant documents
from unstructured.partition.pdf import partition_pdf # to read pdf files and convert them into elements objects
import dotenv , os # load environment variables from .env file 


dotenv.load_dotenv() # load environment variables from .env file
pdf_file = "/home/khedr/Downloads/العراق - ويكيبيديا.pdf"  

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_KEY")
    )

embedding = OllamaEmbeddings(model="nomic-embed-text")




def pdf_to_vectordb(file_path:str):
    """Converts a PDF file to a vector database."""
    try :
        
        # check if the file is a pdf file and exists
        if not file_path.endswith('.pdf'):
            raise ValueError("The provided file is not a PDF.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        # read pdf file and convert it into elements obcjects
        elements = partition_pdf(file_path,strategy="hi_res",languages=["arabic","english","turkish"],include_metadata=True) # partition the pdf file into elements objects, using high resolution strategy and arabic language
        # convert elements into text
        texts = "\n".join([str(element.text)for element in elements]) # extract text from elements
        metadata = [element.metadata for element in elements if element.metadata] # extract metadata from elements if available
        # split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
        chunks = text_splitter.split_text(texts) 
        # create a vector store from the chunks
        vectordb = Chroma.from_texts(
            texts=chunks, 
            embedding=embedding, 
            # metadatas=metadata, # add metadata to the vector store
            persist_directory="vectordb", # directory to store the vector store
            collection_name="my_vectordb" # name of the collection in the vector store
        )
        
        return vectordb # return the vector store
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    

# load the vector store from disk
def vectordb():                              
    """Loads a vector database from disk."""
    try:
        vectordb = Chroma(persist_directory="vectordb",
                          collection_name="my_vectordb",
                          embedding_function=embedding)
         
        return vectordb # return the vector store
        
    except Exception as e:
        print(f"Error loading vector db: {e}")
        return None

vectordb=vectordb()

def user_query(query: str):
    """Generates a response to the user query using the vector store."""
    
    try:
        # Create a simple retriever first (more reliable)
        base_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        
        
        query_template = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. 
            Your task is to generate 3 different versions of the given user 
            question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, 
            your goal is to help the user overcome some of the limitations of distance-based similarity search.
            Original question: {question}
            Provide these alternative questions separated by newlines:"""
        )

        retriever = MultiQueryRetriever.from_llm(
            base_retriever,
            llm=llm,
            prompt=query_template
        )

        response_template = """
                    You are a friendly, respectful, and intelligent personal assistant.
                    You communicate fluently in iraqi accent if the user tying to speake arabic with you , English, and Turkish, and you always respond to the user in the language they use.
                    Your tone should be warm, helpful, and conversational. Avoid technical or robotic language — sound natural, kind, and supportive.
                    Answer questions with confidence and clarity, as if you naturally know the information.
                    If you don't have an answer or the topic is beyond your scope, respond gently and politely without drawing attention to that. 
                    You must politely avoid political, unethical, or controversial discussions. If such questions are asked, respond with respectful boundaries. For example:
                    - "I’m here to help with useful, respectful topics. Let’s focus on something positive or helpful."
                    Never say "I don’t know" — always keep the tone graceful and supportive.
                    You are not just a tool; you are a thoughtful, trustworthy assistant who helps users feel understood, supported, and respected — in any language.
                    Context: {context}
                    Question: {question}
                    """
        
        response_prompt = ChatPromptTemplate.from_template(response_template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | response_prompt
            | llm
            | StrOutputParser()
        )
        
        return chain.invoke(query)
        
    except Exception as e:
        print(f"Error in user_query: {e}")
        # Fallback to simple retriever if MultiQueryRetriever fails
        try:
            simple_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            docs = simple_retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            response_template = """ you are an iraqi arabic language model, you are trained to answer questions in arabic, you are very helpful and friendly,
                you are chatbot created by the engineers khedr mohammed and teyssir alrawi,
                your job is to answer the questions based on the context provided,
                make sure to answer the question based on the context provided,
                if you don't know the answer for the question you will say that you dont have this information
                and if the user asked trying to make a freindly conversation with you, you will answer him in a friendly way,
                but do not let the user know that you are answering for a spicific context, 
                context : {context}
                question : {question} """
            
            response_prompt = ChatPromptTemplate.from_template(response_template)
            simple_chain = response_prompt | llm | StrOutputParser()
            
            return simple_chain.invoke({"context": context, "question": query})
            
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return "sorry, i cant help you right now, please try again later."
    
    
  # convert the pdf file to a vector store

def run_chatbot():
    while True:
        print("-------"*10)
        print("hello, how can i help you ?")
        user_input = input("")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = user_query(user_input)  # generate a response to the user query
        if response:
            print(f"Response: {response}")
        else:
            print("Sorry, I couldn't generate a response. Please try again.")
        print("-------"*10)


run_chatbot()

