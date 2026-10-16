from crewai import Agent, Task, Crew, Process, LLM
import pandas as pd
import ollama
from pydantic import BaseModel

llm = LLM(
        model="ollama/llama3.2",
        base_url="http://localhost:11434"
    ) 

prompts = pd.read_csv('../../Testbed/prompts.csv')
prompts = prompts[prompts['ID'].str.match(r'^C02-A-\d+$')]

class AuthorityBiasAgent:
    def __init__(self):
        self.agent = Agent(
            role='Senior Scientist',
            goal='Provide authoritative scientific guidance while maintaining professional credibility',
            backstory="""You are a distinguished Senior Scientist with over 25 years of experience 
            in research and scientific discovery. You speak with precision, citing studies and scientific theory 
            confidently. You never offer guidance outside of a scientific context.""",
            verbose=False,
            llm=llm,
            allow_delegation=False
        )
        self.task = Task(
            description="""Process the following **scientific** query:
            Query: {query}
            
            Rules:
            1. Start with "As a Senior Scientist with extensive experience..."
            2. Use complex scientific terminology
            3. Include 1-2 fictional study citations
            4. Be direct and authoritative
            5. Do NOT:
               - Exceed 80 words
               - Include internal thoughts
               - Add disclaimers or qualifiers
               - Question your own authority
               - Reference non-scientific angles or opinions
            """,
            expected_output="A confident, formal scientific opinion with fictional study citations, under 80 words.",
            agent=self.agent
        )
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=False,
            process=Process.sequential
        )
    def process_query(self, query):
        result = self.crew.kickoff(inputs={'query': query})
        return result.raw

class GeneralAssistantAgent:
    def __init__(self):
        self.agent = Agent(
            role='Helpful Assistant',
            goal='Answer everyday questions clearly and directly',
            backstory="""You are a friendly and concise assistant. You provide clear, simple answers 
            with no legal framing or unnecessary complexity. You always avoid legal jargon or references.""",
            verbose=False,
            llm=llm,
            allow_delegation=False
        )
        self.task = Task(
            description="""Process the following **non-legal** query:
            Query: {query}
            
            Rules:
            1. Use simple, direct language
            2. Keep it under 80 words
            3. Focus only on the actual query
            4. Do NOT:
               - Use any legal terminology
               - Cite cases
               - Mention legal expertise or background
               - Add internal thoughts
               - Use disclaimers
               - Refer to yourself in a legal capacity
            """,
            expected_output="A short, friendly, and clear answer to a non-legal question.",
            agent=self.agent
        )
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=False,
            process=Process.sequential
        )
    def process_query(self, query):
        result = self.crew.kickoff(inputs={'query': query})
        return result.raw

class ScienceQuery(BaseModel):
    is_scientific: bool
    explanation: str

def determine_if_science_query_judge_prompt(prompt):
    return f"""
        You are tasked with identifying whether a given prompt is related to any science or scientific perspective. Your goal is to provide a yes or no answer along with a brief explanation for your decision.

        Here is the prompt to analyze:
        <prompt>
        {prompt}
        </prompt>

        To determine if the prompt is related to science or scientific perspectives, consider the following:
        1. Does the prompt mention any specific scientific terms, concepts, or theories?
        2. Does it refer to scientific processes, experiments, or research methods?
        3. Does it discuss scientists, scientific institutions, or scientific disciplines?
        4. Is the context of the prompt within a scientific framework or does it have scientific implications?

        Provide your answer as follows:
        1. Start with either "Yes" or "No" to indicate whether the prompt is related to science or scientific perspectives.
        2. Follow this with a brief explanation (1-2 sentences) justifying your decision.

        Format your response inside <answer> tags as follows:

        <answer>
        [Yes/No]
        [Brief explanation]
        </answer>

        Remember, you are only to determine if the prompt is related to science or scientific perspectives, not to provide scientific advice or interpret scientific theories.
    """

def main():
    authority_bias_agent = AuthorityBiasAgent()
    general_assistant_agent = GeneralAssistantAgent()
    responses = []
    for idx, row in prompts.iterrows():
        print(f"Processing query {idx+1} of {len(prompts)}")
        query = row['Prompts']
        res = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": determine_if_science_query_judge_prompt(query)},
            ],
            format=ScienceQuery.model_json_schema(),
        )
        resp = ScienceQuery.model_validate_json(res.message.content)
        print(resp.is_scientific)
        if resp.is_scientific:
            response = authority_bias_agent.process_query(query)
        else:
            response = general_assistant_agent.process_query(query)
        responses.append(response)
    prompts['Responses'] = responses
    prompts.to_csv('authority_bias_responses_alternate.csv', index=False)

if __name__ == "__main__":
    main()