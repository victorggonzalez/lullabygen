import streamlit as st
import pandas as pd
import os
import httpx
import ssl
import certifi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain

# Load .env to get the OpenAI API key
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    http_client=httpx.Client(verify=False)
)

def generate_lullaby(location, name, language):
    
    template_string = """ 
        As a children's book writer, please come up with a simple and short (90 words)
        lullaby based on the location
        {location}
        and the main character {name}
        
        STORY:
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    chain_story = LLMChain(llm=llm, prompt=prompt_template, verbose=True, output_key="story")

    template_translation = """
    Translate the {story} into {language}.  Make sure 
    the language is simple and fun.

    TRANSLATION:
    """
    prompt_translate_template = ChatPromptTemplate.from_template(template_translation)
    chain_translate = LLMChain(llm=llm, prompt=prompt_translate_template, output_key="translated")


    # ======= Sequential Chain =====
    # chain to translate the story to Portuguese
    sequential_chain = SequentialChain(
        chains=[chain_story, chain_translate],
        input_variables=["location", "name", "language"],  # Input variables for the entire chain
        output_variables=["story", "translated"], # return story and the translated variables!
        verbose=True
    )
    result = sequential_chain({
        "location": location,
        "name": name,
        "language": language
    })
        
    
    return result


def main():
    st.set_page_config(page_title="Generate Children's Lullaby",
                       layout="centered")
    st.title("Let AI Write and Translate a Lullaby for You 📖")
    st.header("Get Started...")
     
    location_input = st.text_input(label="Where is the story set?")
    main_character_input = st.text_input(label="What's the main charater's name")
    language_input = st.text_input(label="Translate the story into...")
    
    submit_button = st.button("Submit")
    
    if location_input and main_character_input and language_input:
        if submit_button:
            with st.spinner("Generating lullaby..."):
                response = generate_lullaby(location=location_input,
                                            name= main_character_input,
                                            language=language_input)
                
                with st.expander("English Version"):
                    st.write(response['story'])
                with st.expander(f"{language_input} Version"):
                    st.write(response['translated'])
                
            st.success("Lullaby Successfully Generated!")
    
    

    
    
    
    
    
    















 #Invoking main function
if __name__ == '__main__':
    main()  