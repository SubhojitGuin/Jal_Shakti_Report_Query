import streamlit as st

# Function to generate chatbot responses
def chatbot_response(user_input):
    # You can modify this to integrate with any chatbot model or logic
    response = f"Chatbot: You said '{user_input}'. How can I help you more?"
    return response

# Streamlit app interface
def main():
    st.title("Simple Chatbot Interface")

    # Store conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Input text box for user query
    user_input = st.text_input("You: ", key="user_input")

    if st.button("Send"):
        if user_input:
            # Add user input and bot response to conversation history
            st.session_state.conversation.append(f"You: {user_input}")
            bot_response = chatbot_response(user_input)
            st.session_state.conversation.append(bot_response)
    
    # Display conversation history
    for message in st.session_state.conversation:
        st.write(message)

if __name__ == "__main__":
    main()
