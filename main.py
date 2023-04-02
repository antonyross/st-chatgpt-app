# chatgpt_app.py




####### GENERATIVE QUESTION ANSWERING #######

import openai
import tiktoken
import pandas as pd
import numpy as np
import streamlit as st
import toml


COMPLETIONS_MODEL = "gpt-3.5-turbo-0301"
EMBEDDING_MODEL = "text-embedding-ada-002"


# Get context from DataFrame; Set index to (Document, Section) tuple
df = pd.read_csv('ML_module_text.csv')
df = df.set_index(["Document", "Section"])


# Get the embedding of a text
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


# Convert document sections to embeddings
def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the (Document, Section) tuple that it corresponds to.
    """
    return {
        index: get_embedding(row.Content) for index, row in df.iterrows()
    }



#document_embeddings = compute_doc_embeddings(df)



# Find the most similar document embeddings to the question embedding
def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, 
    the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


# Order the document embeddings by similarity to the question embedding
def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    1.) Find the embedding for the supplied query
    2.) Compare it against all of the pre-calculated document (sections) embeddings
    3.) Find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


# Add the most relevant document sections to the query prompt
    """
    Once we've calculated the most relevant pieces of context, 
    we construct a prompt by simply prepending them to the supplied query. 
    It is helpful to use a query separator to help the model distinguish 
    between separate pieces of text.
    """
# The maximum number of tokens, collectively, that a returned context can contain
MAX_SECTION_LEN = 800
SEPARATOR = "\n* "                  # Context separator contains 3 tokens
ENCODING = "cl100k_base"            # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant context
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
           # section_index is the (Document, Section) tuple
    for _, section_index in most_relevant_document_sections:
        
        # document_section is the row associated with the given (Document, Section) index
        document_section = df.loc[section_index]
        
        # Add contexts until we run out of space.
        chosen_sections_len += document_section.Num_Tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.Content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
      
    response_when_unable_to_answer = "The current module doesn't provide enough context to answer that"
    
    header = f"""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, respond "{response_when_unable_to_answer}."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"



def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301", 
                messages=[{"role": "user", "content": prompt}],
                temperature = 0
            )

    return response.choices[0].message.content.strip(" \n")



def main():
    '''
    This function gets the user input, passes it to the ChatGPT function and 
    displays the response
    '''
    
    # Get OpenAI API key
    with open('secrets.toml', 'r') as f:
        config = toml.load(f)
    openai.api_key = config['OPENAI_KEY']
    
    
    # Convert document sections to embeddings
    document_embeddings = compute_doc_embeddings(df)
    
    st.title("ChatGPT with SDO")
    st.sidebar.header("Instructions")
    st.sidebar.info(
       '''This is a web application that enables you to ask questions related to any module. 
       Enter a **query** in the **text box** and **press 'Submit'** to receive 
       a **response** from ChatGPT.
       '''
    )
    
    # Create text area widget to receive a question
    question = st.text_area(f"Ask me about the ML module:")
                 
    # Get answer to question...
    if st.button("Submit"):
        with st.spinner("Generating response..."):    
            response = answer_query_with_context(question, df, document_embeddings)
        
        # Create text area widget to provide a response to a question
        st.text_area("Response:", value=response, height=None)
    
    return 


main()
    







#if __name__ == "__main__":
    
    
    
    
    
    
    
    
    
    
