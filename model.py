#from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers 
from langchain.chains import RetrievalQA
import chainlit as cl    
import sqlite3
from twilio.rest import Client
import re 
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

db_faiss_path_general = 'vectorstore/db_faiss_general'
db_faiss_path_emergency = 'vectorstore/db_faiss_emergency'

custom_prompt_template = """
You are a medical assistant. Answer the following question based on the context provided.

**Context**: {context}
**Question**: {question}

Provide a detailed and concise answer related to the user's question. Avoid repeating information and keep the response focused.

Format your response as follows:
Answer: [Provide a detailed response to the question]

- Disease/Condition: [Disease Name]
- Medical Specialty: [Specialty Name]

Note: [Provide additional relevant information if needed.]
"""






def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables =['context', 'question'])
    return prompt


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


#Loading the model :

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm 
    
#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(db_faiss_path_general, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#contacting specialist:
def get_specialist_by_specialty(specialty):
    conn = sqlite3.connect('specialists.db')
    cursor = conn.cursor()

    # Find a specialist based on specialty and availability
    cursor.execute('''
        SELECT name, phone, availability FROM specialists
        WHERE specialty = ? AND availability = 'Available'
        LIMIT 1
    ''', (specialty,))

    specialist = cursor.fetchone()
    conn.close()
    
    if specialist:
        return {
            'name': specialist[0],
            'phone': specialist[1],
            'availability': specialist[2]
        }
    else:
        return None

def sanitize_response(text):
    """
    Cleans up repetitive or garbled text in the bot's response.
    """
    # Basic cleanup to remove repeated words and extra spaces
    cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # Removes repeated words
    cleaned_text = re.sub(r'(\w+\s*,\s*){2,}', '', cleaned_text)  # Removes repeating phrases
    return cleaned_text


#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response    
    
#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = """welcome to the Medical Bot. I understand you're in a difficult situation, "
        and I'm here to support you. Please let me know how I can help. What is your query?"""
    await msg.update()

    cl.user_session.set("chain", chain)    
        
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain instance
    chain = cl.user_session.get("chain") 

    # Define the callback handler with streaming enabled
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # Get the response from the chatbot
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]

    # Sanitize the output by removing repetitive patterns
    answer = sanitize_response(answer)

    # Regex pattern to extract the medical specialty and disease
    pattern_specialty = r"Medical Specialty:\s*(.*)"
    
    # Search for the medical specialty and disease using the pattern
    match_specialty = re.search(pattern_specialty, answer)
    
    # Twilio credentials (replace with your actual credentials)
    account_sid = os.getenv("my_account_sid")
    auth_token = os.getenv("my_auth_token")
    client = Client(account_sid, auth_token)

    if match_specialty:
        
        medical_specialty = match_specialty.group(1).strip()
        message_specialist = f"Alert: A patient needs assistance related to {medical_specialty}. Please contact back if available."
        sms_specialist = client.messages.create(
     body=message_specialist,
     from_='my_twilio_number',  # Your Twilio phone number
     to='my_phone_number'  # The specialist's phone number from the database
)
        final_answer = f"{answer}\n\nWe'll contact the relevant specialist in {medical_specialty}."
        print(sms_specialist.sid)
    else:
        message_general = f"Urgent: A patient requires attention to identify the right disease. Please contact the patient at your earliest convenience."
        sms_general = client.messages.create(
     body=message_general,
     from_='my_twilio_number',  # Your Twilio phone number
     to='my_phone_number'  # The specialist's phone number from the database
)
        final_answer = f"{answer}\n\nWe'll contact a health provider to follow up."
        print("Medical Specialty not found.")
        print(sms_general.sid)

    # Send the final response back to the user
    await cl.Message(content=final_answer).send()

    
  