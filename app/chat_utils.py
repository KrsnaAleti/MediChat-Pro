from euriai.langchain import create_chat_model # Import the function to create a chat model - this is a wrapper around Langchain's ChatOpenAI built by EURON

def get_chat_model(api_key: str):
    """
    Initializes and returns a chat model configured with a specific API key, model name, and temperature.

    This function uses the `create_chat_model` function from the `euriai.langchain` library,
    which is a wrapper around Langchain's ChatOpenAI, to set up a conversational AI model.

    Args:
        api_key (str): The API key required to authenticate with the chat model service.

    Returns:
        A chat model instance configured with the specified parameters.
    """
    return create_chat_model(api_key=api_key, 
                             model="gpt-4.1-nano", 
                             temperature=0.7)
    
def ask_chat_model(chat_model, question: str):
    """
    Sends a question to the provided chat model and returns the response.

    Args:
        chat_model: An instance of a chat model created using `get_chat_model`.
        question (str): The question to be sent to the chat model.

    Returns:
        str: The response from the chat model.
    """
    response = chat_model.invoke(question)
    return response.content