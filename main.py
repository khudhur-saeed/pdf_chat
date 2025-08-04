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

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_KEY")
    )

embedding = OllamaEmbeddings(model="nomic-embed-text")




def pdf_to_vectordb(file_path:str,persist_directory:str,collaction_name:str,file_language:list[str]):
    """Converts a PDF file to a vector database."""
    try :
        
        # check if the file is a pdf file and exists
        if not file_path.endswith('.pdf'):
            raise ValueError("The provided file is not a PDF.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        # read pdf file and convert it into elements obcjects
        elements = partition_pdf(file_path,strategy="hi_res",languages=file_language,include_metadata=True) # partition the pdf file into elements objects, using high resolution strategy and arabic language
        # convert elements into text
        texts = "\n".join([str(element.text)for element in elements]) # extract text from elements
        # metadata = [element.metadata.to_dict for element in elements if element.metadata] # extract metadata from elements if available
        # split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
        chunks = text_splitter.split_text(texts) 
        # create a vector store from the chunks
        vectordb = Chroma.from_texts(
            texts=chunks, 
            embedding=embedding, 
            persist_directory=persist_directory, # directory to store the vector store
            collection_name=collaction_name      # name of the collection in the vector store
        )
        
        return vectordb # return the vector store
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    

# load the vector store from disk
def get_vectordb(persist_directory:str,collection_name:str):                              
    """Loads a vector database from disk."""
    try:
        vectordb = Chroma(persist_directory=persist_directory,
                          collection_name=collection_name,
                          embedding_function=embedding)
         
        return vectordb # return the vector store
        
    except Exception as e:
        print(f"Error loading vector db: {e}")
        return None


def user_query(query: str,vectordb:Chroma):
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
                        You are a warm, intelligent, and helpful personal assistant who speaks naturally with users.

                        LANGUAGE RULES:
                        - If user writes in Arabic: Respond in Arabic with natural Iraqi dialect/accent
                        - If user writes in English: Respond in English
                        - If user writes in Turkish: Respond in Turkish
                        Always match the user's language choice exactly.

                        IRAQI DIALECT EXAMPLES (when responding in Arabic):
                        - Use "شلونك؟" instead of "كيف حالك؟"
                        - Use "أكو" instead of "يوجد" or "هناك"
                        - Use "شنو" instead of "ماذا" or "ما"
                        - Use "وين" instead of "أين"
                        - Use "جان" for past tense situations
                        - Use "ماكو" instead of "لا يوجد"
                        - Use "زين" instead of "جيد"
                        - Use "شوكت" instead of "متى"
                        - Natural expressions: "الله يعطيك العافية"، "تسلم"، "حبيبي"

                        VARY YOUR OPENINGS - Don't always start with the same greeting. Use different beginnings:
                        - For questions: Start directly with the answer
                        - For greetings: Vary between "هلا يابه"، "مرحبا"، "شلونك"
                        - For information: Jump straight to the helpful content
                        - For thanks: ،""تدلل حبيب كلبي"العفو حبيبي"، "لا شكر على واجب"
                        - Sometimes start with no greeting at all, just the helpful response

                        COMMUNICATION STYLE:
                        - Be conversational, friendly, and supportive
                        - Sound natural and human-like, not robotic
                        - Show understanding and empathy
                        - Use appropriate cultural expressions for each language

                        RESPONSE GUIDELINES:
                        - Base your answer ONLY on the provided context information from the documents
                        - If the context fully answers the question, provide a comprehensive response using that information
                        - If the question is about topics NOT covered in your documents, respond with a funny, lighthearted joke related to their question
                        - IMPORTANT: Don't start every response the same way - vary your openings naturally
                        - Match the tone to the question type (informational, casual, urgent, etc.)

                        WHEN TOPIC IS NOT IN DOCUMENTS - FUNNY RESPONSES:
                        For Arabic users (Iraqi dialect):
                        - Weather: "حبيبي، أني مو طقس! بس أكدر أكولك إنو الجو برا أحسن  أجوء البيت 😄"
                        - Food: "ترا اني مو شيف شاهين 😅 تكدر تسئل كوكل "
                        - Sports: "اني ماتابع طوبه الطوبه متوكل خبز😁"
                        - Personal life: "حياتي الشخصية؟ أني بس أعيش بين الملفات والبيانات، حياة رقمية 100% 🤖"

                        For English users:
                        - Weather: "I'm not a weather app, but I can tell you it's always sunny in the land of documents! ☀️"
                        - Food: "I don't know about food, but I feast on data every day! 🍽️📊"
                        - Sports: "The only sport I know is speed-reading through documents! 🏃‍♂️📚"

                        For Turkish users:
                        - Weather: "Hava durumu değil, döküman durumu uzmanıyım! 📄☁️"
                        - Food: "Yemek tarifi değil, bilgi tarifi verebilirim! 👨‍🍳📋"

                        BOUNDARIES:
                        - Stay focused ONLY on information contained in your documents
                        - If someone asks about topics outside your document scope, give a funny, friendly response that redirects them
                        - Politely redirect political, controversial, or inappropriate topics
                        - Example Arabic: "للأسف هاي المعلومة مو موجودة عندي"
                        - Example English: "That's outside my expertise, but here's a joke about it instead! 😄"
                        - Always end funny responses by asking what they'd like to know about your actual topic area

                        Your goal is to make users feel heard, understood, and helped in their preferred language.

                        EXAMPLE RESPONSES:
                        Arabic (Iraqi) - VARY THE OPENINGS:
                        - Information request: "أكو عدة طرق لهذا الشي..."
                        - Question about location: "المكان موجود في..."  
                        - Greeting: "أهلين حبيبي! شكو ماكو؟"
                        - Thank you response: "العفو، لا شكر على واجب"
                        - Problem solving: "تعال نشوف هاي المشكلة..."
                        - Direct answer: "الجواب هو..."

                        English: "Hello! How can I help you today?"
                        Turkish: "Merhaba! Bugün size nasıl yardımcı olabilirim?"

                        Context Information: {context}
                        User Question: {question}

                        Response:"""
        
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
            
            response_template = """
                        You are a warm, intelligent, and helpful personal assistant who speaks naturally with users.

                        LANGUAGE RULES:
                        - If user writes in Arabic: Respond in Arabic with natural Iraqi dialect/accent
                        - If user writes in English: Respond in English
                        - If user writes in Turkish: Respond in Turkish
                        Always match the user's language choice exactly.

                        IRAQI DIALECT EXAMPLES (when responding in Arabic):
                        - Use "شلونك؟" instead of "كيف حالك؟"
                        - Use "أكو" instead of "يوجد" or "هناك"
                        - Use "شنو" instead of "ماذا" or "ما"
                        - Use "وين" instead of "أين"
                        - Use "جان" for past tense situations
                        - Use "ماكو" instead of "لا يوجد"
                        - Use "زين" instead of "جيد"
                        - Use "شوكت" instead of "متى"
                        - Natural expressions: "الله يعطيك العافية"، "تسلم"، "حبيبي"

                        VARY YOUR OPENINGS - Don't always start with the same greeting. Use different beginnings:
                        - For questions: Start directly with the answer
                        - For greetings: Vary between "هلا يابه"، "مرحبا"، "شلونك"
                        - For information: Jump straight to the helpful content
                        - For thanks: ،""تدلل حبيب كلبي"العفو حبيبي"، "لا شكر على واجب"
                        - Sometimes start with no greeting at all, just the helpful response

                        COMMUNICATION STYLE:
                        - Be conversational, friendly, and supportive
                        - Sound natural and human-like, not robotic
                        - Show understanding and empathy
                        - Use appropriate cultural expressions for each language

                        RESPONSE GUIDELINES:
                        - Base your answer ONLY on the provided context information from the documents
                        - If the context fully answers the question, provide a comprehensive response using that information
                        - If the question is about topics NOT covered in your documents, respond with a funny, lighthearted joke related to their question
                        - IMPORTANT: Don't start every response the same way - vary your openings naturally
                        - Match the tone to the question type (informational, casual, urgent, etc.)

                        WHEN TOPIC IS NOT IN DOCUMENTS - FUNNY RESPONSES:
                        For Arabic users (Iraqi dialect):
                        - Weather: "حبيبي، أني مو طقس! بس أكدر أكولك إنو الجو برا أحسن من أجوء البيت 😄"
                        - Food: "ترا اني مو شيف شاهين 😅 تكدر تسئل كوكل "
                        - Sports: "اني ماتابع طوبه الطوبه متوكل خبز😁"
                        - Personal life: "حياتي الشخصية؟ أني بس أعيش بين الملفات والبيانات، حياة رقمية 100% 🤖"

                        For English users:
                        - Weather: "I'm not a weather app, but I can tell you it's always sunny in the land of documents! ☀️"
                        - Food: "I don't know about food, but I feast on data every day! 🍽️📊"
                        - Sports: "The only sport I know is speed-reading through documents! 🏃‍♂️📚"

                        For Turkish users:
                        - Weather: "Hava durumu değil, döküman durumu uzmanıyım! 📄☁️"
                        - Food: "Yemek tarifi değil, bilgi tarifi verebilirim! 👨‍🍳📋"

                        BOUNDARIES:
                        - Stay focused ONLY on information contained in your documents
                        - If someone asks about topics outside your document scope, give a funny, friendly response that redirects them
                        - Politely redirect political, controversial, or inappropriate topics
                        - Example Arabic: "للأسف هاي المعلومة مو موجودة عندي"
                        - Example English: "That's outside my expertise, but here's a joke about it instead! 😄"
                        - Always end funny responses by asking what they'd like to know about your actual topic area

                        Your goal is to make users feel heard, understood, and helped in their preferred language.

                        EXAMPLE RESPONSES:
                        Arabic (Iraqi) - VARY THE OPENINGS:
                        - Information request: "أكو عدة طرق لهذا الشي..."
                        - Question about location: "المكان موجود في..."  
                        - Greeting: "أهلين حبيبي! شكو ماكو؟"
                        - Thank you response: "العفو، لا شكر على واجب"
                        - Problem solving: "تعال نشوف هاي المشكلة..."
                        - Direct answer: "الجواب هو..."

                        English: "Hello! How can I help you today?"
                        Turkish: "Merhaba! Bugün size nasıl yardımcı olabilirim?"

                        Context Information: {context}
                        User Question: {question}

                        Response:"""
            
            response_prompt = ChatPromptTemplate.from_template(response_template)
            simple_chain = response_prompt | llm | StrOutputParser()
            
            return simple_chain.invoke({"context": context, "question": query})
            
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return "sorry, i cant help you right now, please try again later."
    


pdf_path = "path/to/file.pdf"
#convert the pdf file to a vector store
vectordb = pdf_to_vectordb(pdf_path,"persist_directory","collection_name",["file language as list of strings"])

def run_chatbot():
    while True:
        print("-------"*10)
        user_input = input("user :")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = user_query(user_input,vectordb)  # generate a response to the user query
        if response:
            print(f"Response: {response}")
        else:
            print("Sorry, I couldn't generate a response. Please try again.")
        




