from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv


load_dotenv()

os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
os.environ['PINECONE_API_KEY']=os.getenv('PINECONE_API_KEY')

###Function to  retuen respose from Gemini Pro
def model_response(file,query):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    context="\n\n".join(str(p.page_content) for p in file)
    chunks=text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    searcher=FAISS.from_texts(chunks,embeddings).as_retriever()

    q='Which education people are having income >50k'
    records=searcher.get_relevant_documents(q)
    print(records)

    promt_template="""

     You have to answer from the Provided Context  and Make sure that you Provided all the details\n

     context :{context}\n

     Question :{question}\n

     Answer:

"""


    prompt= PromptTemplate(template=promt_template,input_variables=["context","question"])

    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.9)

    chains=load_qa_chain(model,chain_type="stuff", prompt=prompt)

    result=chains(
        {
            "input_documents":records,
            "question":query
        }
        ,return_only_outputs=True
    )

    return result['output_text']






