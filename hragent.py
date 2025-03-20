import asyncio
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents.azure_ai import AzureAIAgent,AzureAIAgentSettings
import asyncio
import os
from dotenv import load_dotenv

os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"] = "gpt-4o"
MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")

os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"] = "eastus.api.azureml.ms;a182fe30-ddfd-4f86-9bb2-9278c2f0c684;rg-nithin-9486_ai;nithin-8183"
PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")

async def main() -> None:
    # Use asynchronous context managers to authenticate and create the client.

    ai_agent_settings = AzureAIAgentSettings(
        model_deployment_name=MODEL_DEPLOYMENT_NAME,
        project_connection_string=PROJECT_CONNECTION_STRING
        #endpoint=os.getenv("azure_endpoint"),  # Required
        #subscription_id=os.getenv("AZURE_AI_AGENT_SUBSCRIPTION_ID"),  # Required
        #resource_group_name=os.getenv("AZURE_AI_AGENT_RESOURCE_GROUP_NAME"),  # Required
        #project_name=os.getenv("project_name"),  # Required
    )

    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds,settings=ai_agent_settings) as client,
    ):
        # 1. Retrieve the agent definition based on the assistant ID.
        #    Replace "asst_MwpyijFo7T4MEzwvS8Pb5F98" with your actual assistant ID.
        agent_definition = await client.agents.get_agent(
            agent_id="asst_g385wUu9Xbd1vbJtTllRhQ1B",
        )

        # 2. Create a Semantic Kernel agent using the retrieved definition.
        agent = AzureAIAgent(client=client, definition=agent_definition)

        # 3. Create a new conversation thread.
        thread = await client.agents.create_thread()

        try:
            # 4. Define the single query and add it as a chat message.
            user_query = "Hi, I am Tharun Ganesh, help me to find the suitable job roles(only the role names)"
            await agent.add_chat_message(thread_id=thread.id, message=user_query)
            print(f"# User: {user_query}")

            # 5. Retrieve and print the agent's response.
            response = await agent.get_response(thread_id=thread.id)
            print(f"# Agent: {response}")
        finally:
            # 6. Cleanup: Delete the conversation thread.
            await client.agents.delete_thread(thread.id)
            # Note: The agent is not deleted so it can be reused later.

if __name__ == "__main__":
    asyncio.run(main())