Objective:
Create a simple chatbot application using any suitable open-source LLM model. The chatbot should provide a basic ChatGPT-like chat experience and maintain conversation history on a session basis.

Requirements:

LLM Integration
- Load and integrate any suitable open-source LLM model.
- The model can be selected based on available system resources.
- Implement a basic prompt/template to generate appropriate responses.

Chat UI
- Create a simple, clean chat interface similar to ChatGPT.
- Display user and assistant messages clearly.
- Allow users to send multiple messages within the same conversation session.

Session-Based Chat History
- Maintain chat history separately for each user/session.
- Previous messages should be passed as context to the LLM when generating new responses.
- Starting a new session should create a fresh conversation without using history from previous sessions.
- Context-Aware Conversation
- The chatbot should understand follow-up questions based on previous conversation context.

For example:
User: "What is RAG?"
User: "What are its main components?"
User: "Can you explain the second one in detail?"
- The chatbot should understand that the second and third questions are related to the previous conversation rather than treating them as independent questions.
- Basic Response Handling
- Handle normal conversational questions appropriately.
- Maintain the context of the conversation throughout the session.
- Handle cases where the user changes the topic during the same session.

Expected Outcome:
A working chatbot application with a ChatGPT-like UI, an open-source LLM, session-based conversation history, and proper context retention for follow-up questions.