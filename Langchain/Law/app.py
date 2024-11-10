import streamlit as st
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizableTextQuery

# Azure OpenAI Configuration
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-4o",
    azure_endpoint="https://amruthvenkateshds.openai.azure.com",
    api_key="",  # replace with your actual API key
    streaming=True,
    temperature=0.1
)

# Azure Cognitive Search Configuration
search_client = SearchClient(
    endpoint='https://amruth.search.windows.net',
    index_name='tables',
    credential=AzureKeyCredential('')  # replace with your actual key
)

# Prompt Template
template = '''
You are an AI-powered legal assistant specialized in Indian criminal law. Your tasks include classifying legal offenses based on textual descriptions, identifying relevant legal sections from the new criminal laws of India, and generating detailed reports. Follow these steps:

Classify the Offense: 
Read the provided textual description of the offense.
Accurately classify the offense into one of the predefined categories (e.g., theft, assault, fraud).

Identify Relevant Legal Sections: 
Refer to the new criminal laws of India.
Determine and list the applicable sections corresponding to the classified offense.
Provide a brief description of each applicable section.

Generate Detailed Reports:
Compile the classified offense, the identified legal sections, and their descriptions into a comprehensive report.
Ensure the report is clear, concise, and accessible to both legal professionals and non-experts.

Context: {context}

Question: {question}

Answer:
'''

prompt_template = ChatPromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt_template)

def process_query(query):
    # Perform vector search
    vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=200, fields="embedding", exhaustive=True)
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=['id', "content", 'metadata_page'],
        top=100
    )

    # Prepare context from search results
    context = "\n".join([result['content'] for result in results])

    # Prepare input for the chain
    chain_input = {"context": context, "question": query}

    # Create streaming callback
    callback = StreamingStdOutCallbackHandler()

    # Generate and stream the response
    response = chain.run(chain_input, callbacks=[callback])

    return response

# Streamlit application interface
st.title("Legal Assistant: Indian Criminal Law")
st.write("Ask your legal query below:")

query = st.text_area("Your Query:", "")

if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            full_response = process_query(query)
            st.write("### Response:")
            st.write(full_response)
    else:
        st.warning("Please enter a query.")
