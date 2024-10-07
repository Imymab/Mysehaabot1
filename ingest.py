from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

data_path_general = 'data/general/'

db_faiss_path_general = 'vectorstore/db_faiss_general'

#create vector database: 
def create_vector_db(data_path, db_path, context_type):
    
    #load all pdf files in the directory :
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls = PyPDFLoader)
    documents = loader.load()
    
    #split docs into chunks :
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # Print the type and content of texts for debugging
    print("Types and content of texts:")
    for text in texts:
        print(type(text), text)
    
    #add context type to each document :
    tagged_texts = []
    for doc in texts:
        text = doc.page_content
        if isinstance(text, str):
            tagged_texts.append(Document(page_content=text, metadata={"context":context_type }))
        else:
            print(f"skipping non-string content: {text}")    
    
    #generate embeddings:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device' : 'cpu'})
    db = FAISS.from_documents(tagged_texts, embeddings)
    
    #save your database locally:
    db.save_local(db_path)
    
    
if __name__=='__main__':
    create_vector_db(data_path_general, db_faiss_path_general, 'General')
     


 