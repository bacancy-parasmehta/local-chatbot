"""
Launcher for the chatbot app. On Windows, asyncio's default Proactor event
loop raises an unhandled ConnectionResetError when a client disconnects
abruptly, which crashes the whole server. Switching to the Selector event
loop policy avoids that bug.
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.headless", "true"]
    sys.exit(stcli.main())
