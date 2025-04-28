import os
import streamlit as st
from crewai import Agent, Task, Crew
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def set_openai_api_key(api_key):
    """Set the OpenAI API key as an environment variable."""
    os.environ["OPENAI_API_KEY"] = api_key
    # Also set it for the openai package
    openai.api_key = api_key
    return True

def validate_api_key(api_key):
    """Validate the OpenAI API key by making a simple API call."""
    try:
        # Set the API key
        set_openai_api_key(api_key)
        
        # Try to use the API with a minimal request
        client = openai.OpenAI(api_key=api_key)
        client.models.list(limit=1)
        return True
    except Exception as e:
        st.error(f"API Key validation failed: {str(e)}")
        return False

def run_research_crew(topic):
    """Run the research crew with the given topic."""
    try:
        # Define the agent
        researcher = Agent(
            role="Researcher",
            goal=f"Gather information on {topic} and summarize it",
            backstory="You are an expert researcher skilled at finding and condensing information.",
            verbose=True,
            allow_delegation=False
        )

        # Define the task
        research_task = Task(
            description=f"Research the topic '{topic}' and write a short summary.",
            expected_output=f"A concise summary of {topic}, about 100 words.",
            agent=researcher
        )

        # Create the crew and assign the task
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=True
        )

        # Execute the crew's task
        with st.spinner(f"Researching {topic}... This may take a minute."):
            result = crew.kickoff()
        
        return result
    except Exception as e:
        st.error(f"An error occurred while running the research: {str(e)}")
        return None

# Set up the Streamlit app
st.title("AI Research Assistant")
st.write("This app uses CrewAI to research topics and provide concise summaries.")

# Sidebar for API key input
with st.sidebar:
    st.header("Authentication")
    st.write("Please enter your OpenAI API key to use this application.")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
    
    if api_key:
        if validate_api_key(api_key):
            st.session_state.authenticated = True
            st.success("API Key validated successfully!")
        else:
            st.session_state.authenticated = False
    else:
        st.session_state.authenticated = False
    
    st.write("---")
    st.write("Note: Your API key is only stored temporarily in this session and is not saved anywhere.")

# Main app functionality
if st.session_state.get('authenticated', False):
    st.header("Research a Topic")
    
    # Input for research topic
    topic = st.text_input("Enter a topic to research:", value="Artificial Intelligence Trends in 2025")
    
    if st.button("Start Research"):
        if topic:
            result = run_research_crew(topic)
            if result:
                st.header("Research Summary:")
                st.write(result)
                
                # Option to download the result
                st.download_button(
                    label="Download Summary",
                    data=result,
                    file_name=f"{topic.replace(' ', '_')}_summary.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please enter a topic to research.")
else:
    st.info("Please enter your OpenAI API key in the sidebar to use this application.")
    st.write("This application uses the OpenAI API to generate research summaries. You'll need to provide your own API key to use it.")
    st.write("If you don't have an API key, you can get one from [OpenAI](https://platform.openai.com/account/api-keys).")
