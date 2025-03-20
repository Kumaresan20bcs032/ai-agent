import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential
from semantic_kernel.agents.azure_ai import AzureAIAgent, AzureAIAgentSettings

# Load environment variables
load_dotenv()



os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"] = "gpt-4o"
MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")

os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"] = "eastus.api.azureml.ms;a182fe30-ddfd-4f86-9bb2-9278c2f0c684;rg-nithin-9486_ai;nithin-8183"
PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")

# Initialize FastAPI
app = FastAPI()

# Pydantic model for request validation
class QueryRequest(BaseModel):
    query: str

async def get_ai_agent():
    ai_agent_settings = AzureAIAgentSettings(
        model_deployment_name=MODEL_DEPLOYMENT_NAME,
        project_connection_string=PROJECT_CONNECTION_STRING
    )

    client = AzureAIAgent.create_client(credential=DefaultAzureCredential(), settings=ai_agent_settings)
    agent_definition = await client.agents.get_agent(agent_id="asst_g385wUu9Xbd1vbJtTllRhQ1B")

    return AzureAIAgent(client=client, definition=agent_definition), client

# API Route for Chat Query
@app.post("/chat/")
async def chat_with_agent(request: QueryRequest):
    try:
       
        agent, client = await get_ai_agent()
        thread = await client.agents.create_thread()

        try:
            await agent.add_chat_message(thread_id=thread.id, message=request.query)

            response = await agent.get_response(thread_id=thread.id)
            text_response = response.items[0].text

            return {"query": request.query, "response": text_response}
        

        finally:
            client.agents.delete_thread(thread.id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
