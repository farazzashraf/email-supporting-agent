import httpx
import asyncio
import json
import time

AGENT_URL = "http://localhost:8081"
TENANT_ID = "test_tenant_123"

async def test_agent():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health Check
        print("--- Checking Health ---")
        try:
            res = await client.get(f"{AGENT_URL}/")
            print(f"Status: {res.status_code}")
            print(f"Body: {res.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return

        # 2. Index Knowledge
        print("\n--- Indexing Knowledge ---")
        knowledge = {
            "tenant_id": TENANT_ID,
            "content": "Our support hours are Monday to Friday, 9:00 AM to 6:00 PM EST. We are closed on weekends and public holidays. For urgent matters outside these hours, please email urgent@example.com.",
            "chunk_strategy": "paragraph",
            "metadata": {"source": "manual_test"}
        }
        res = await client.post(f"{AGENT_URL}/index-knowledge", json=knowledge)
        print(f"Status: {res.status_code}")
        print(f"Body: {res.json()}")

        # Give ChromaDB a moment to commit (though it's usually instant)
        time.sleep(1)

        # 3. Chat Request (Testing RAG)
        print("\n--- Sending Chat Request (RAG Test) ---")
        chat_request = {
            "message": "When can I contact your support team?",
            "injected_context": {
                "business_name": "Test Business",
                "faq_text": "We sell high-quality widgets.",
                "tenant_id": TENANT_ID
            }
        }
        res = await client.post(f"{AGENT_URL}/", json=chat_request)
        print(f"Status: {res.status_code}")
        reply = res.json().get("reply", "NO REPLY")
        print(f"Reply: {reply}")

        if "9:00 AM to 6:00 PM" in reply or "Monday to Friday" in reply:
            print("\n[SUCCESS] Agent used the indexed knowledge!")
        else:
            print("\n[FAILURE] Agent might not have used the indexed knowledge.")

        # 4. Upload File Test
        print("\n--- Testing File Upload ---")
        file_content = b"The return policy allows returns within 30 days of purchase with a valid receipt."
        files = {"file": ("return_policy.txt", file_content, "text/plain")}
        data = {"tenant_id": TENANT_ID, "chunk_strategy": "paragraph"}
        res = await client.post(f"{AGENT_URL}/upload-file", data=data, files=files)
        print(f"Status: {res.status_code}")
        print(f"Body: {res.json()}")

        # 5. Chat Request (Testing File RAG)
        print("\n--- Sending Chat Request (File RAG Test) ---")
        chat_request["message"] = "What is your return policy?"
        res = await client.post(f"{AGENT_URL}/", json=chat_request)
        print(f"Status: {res.status_code}")
        reply = res.json().get("reply", "NO REPLY")
        print(f"Reply: {reply}")

        if "30 days" in reply or "receipt" in reply:
            print("\n[SUCCESS] Agent used the knowledge from the uploaded file!")
        else:
            print("\n[FAILURE] Agent might not have used the file knowledge.")

if __name__ == "__main__":
    asyncio.run(test_agent())
