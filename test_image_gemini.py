import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from ai_chat import _get_gemini_key_from_secrets, GEMINI_API_KEY_ENV

api_key = GEMINI_API_KEY_ENV or _get_gemini_key_from_secrets()

# Create a dummy image (1x1 red pixel)
dummy_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82'
b64_img = base64.b64encode(dummy_image).decode('utf-8')

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

message = HumanMessage(
    content=[
        {"type": "text", "text": "What color is this image?"},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64_img}"}
    ]
)

try:
    response = llm.invoke([message])
    print("Direct LLM response:", response.content)
except Exception as e:
    print("Direct LLM error:", e)

from langgraph.prebuilt import create_react_agent
from components.ai_agent import tools

agent = create_react_agent(llm, tools)
try:
    response = agent.invoke({"messages": [message]})
    print("Agent LLM response:", response["messages"][-1].content)
except Exception as e:
    print("Agent error:", e)

