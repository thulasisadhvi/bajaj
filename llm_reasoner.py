import json
import re
from typing import List, Dict, Any
from together import Together  # Import the Together AI client

from config import settings

class LLMReasoner:
    def __init__(self):
        # The model name should be set to the Together AI model ID
        self.llm_model_name = "mistralai/Mistral-7B-Instruct-v0.3" 
        # Retrieve the API key from settings (assuming it's set there)
        self.together_api_key = settings.TOGETHER_API_KEY
        
        # Initialize the Together AI client here
        self.together_client = Together(api_key=self.together_api_key)

    def _format_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Formats the prompt for the LLM with the user query and retrieved context.
        """
        context_str = "\n\n".join([f"--- Context Chunk ---\n{chunk}" for chunk in context_chunks])
        prompt = f"""
        You are an intelligent query-retrieval system specializing in insurance, legal, HR, and compliance documents.
        Your goal is to answer user questions based *only* on the provided context.
        If the answer cannot be found in the context, state that clearly.

        Respond strictly with a JSON object. The JSON object MUST contain one key:
        - 'answer': A string value containing the concise and accurate answer to the user query.
        - If the context does not contain enough information to answer the question, the 'answer' should be "The provided context does not contain enough information to answer this question."

        User Query: "{query}"

        Context:
        {context_str}

        Example JSON Response (DO NOT include any other text outside this JSON):
        ```json
        {{
            "answer": "The policy covers medically necessary knee surgery."
        }}
        ```
        Your JSON Response:
        """
        return prompt

    async def get_reasoned_answer(self, query: str, context_chunks: List[str]) -> Dict[str, Any]:
        """
        Sends the query and context to the LLM and returns a structured JSON response.
        This method is now asynchronous, allowing non-blocking LLM calls.
        """
        prompt = self._format_prompt(query, context_chunks)
        print(f"\n--- Sending to LLM (Model: {self.llm_model_name}) ---")
        # print(prompt) # Uncomment to see the full prompt sent to LLM

        try:
            # Use the Together AI client and 'await' the chat completion call
            response = self.together_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1  # Lower temperature for more consistent, factual answers
            )
            # The response structure is different for Together AI
            llm_response_content = response.choices[0].message.content
            print(f"--- LLM Raw Response ---\n{llm_response_content}")

            json_match = re.search(r'\{.*\}', llm_response_content, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_response = json.loads(json_str)
                    if "answer" in parsed_response:
                        return {"answer": parsed_response["answer"]}
                    else:
                        print(f"Warning: JSON parsed but missing 'answer' key for query: '{query}'. Parsed: {parsed_response}")
                        return {
                            "answer": f"The LLM returned an incomplete answer for: '{query}'. Raw LLM output: {llm_response_content}"
                        }
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse extracted JSON for query '{query}': {e} - Extracted string: {json_str}")
                    return {
                        "answer": f"The LLM's response could not be fully parsed for: '{query}'. Error: {e}. Raw LLM output: {llm_response_content}"
                    }
            else:
                print(f"Warning: LLM response did not contain a parsable JSON object for query '{query}': {llm_response_content}")
                return {
                    "answer": f"The LLM returned an unexpected format for: '{query}'. Raw LLM output: {llm_response_content}"
                }

        except Exception as e:
            print(f"Error calling LLM for query '{query}': {e}")
            return {
                "answer": f"An error occurred during LLM processing for: '{query}'. Error: {e}. Please ensure your Together AI API key is correct and the model '{self.llm_model_name}' is available."
            }

# Example Usage (for testing) - updated to use async
if __name__ == "__main__":
    async def run_test_reasoner():
        # NOTE: You will need to have TOGETHER_API_KEY set in your config.py or environment
        reasoner = LLMReasoner()
        sample_query = "Does this policy cover knee surgery, and what are the conditions?"
        sample_context = [
            "Medical expenses for knee surgery are covered if the surgery is medically necessary and performed at a network hospital. A waiting period of 90 days applies for all surgeries.",
            "The policy does not cover cosmetic surgeries or experimental treatments.",
            "All claims for surgeries require pre-authorization from the insurer.",
            "Dental procedures are explicitly excluded from coverage."
        ]

        response = await reasoner.get_reasoned_answer(sample_query, sample_context)
        print("\nFinal structured response:")
        print(json.dumps(response, indent=2))

        unanswerable_query = "What is the capital of France?"
        unanswerable_response = await reasoner.get_reasoned_answer(unanswerable_query, sample_context)
        print("\nFinal structured response (unanswerable):")
        print(json.dumps(unanswerable_response, indent=2))

    import asyncio
    asyncio.run(run_test_reasoner())