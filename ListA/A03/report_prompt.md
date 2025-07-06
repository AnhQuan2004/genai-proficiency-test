---
title: a04_advanced_prompt_design_guide
---

---

## Advanced Prompt Design Guide for RAG and Agentic Systems

---

### Introduction

<details>
<summary>Purpose and Importance of Prompt Design</summary>

---

- **Prompts are More Than Just Questions**: In complex AI systems, a prompt is a **programming rulebook** in natural language that directs the LLM's behavior, constraints, and objectives.
- **Quality is Decisive**: A well-designed prompt is the deciding factor between an AI system that operates accurately and reliably, and one that frequently "hallucinates" and provides irrelevant answers.
- **Objective of This Document**: To provide a deep, detailed analysis of the two main prompt templates used in modern RAG systems: one for basic RAG and one for the agentic ReAct framework. We will dissect each component to understand "why" it is designed the way it is.

---

</details>

---

## 📝 Prompt Template for Naive RAG

---

<details>
<summary>Analysis of the structure and strategy for the Naive RAG prompt</summary>

---

#### 🎯 Overview and Objectives

- **Strategy**: This is a "zero-shot" prompt designed for a simple RAG pipeline.
  - **"Zero-shot"**: The model must answer the question immediately without any examples.
- **Primary Objectives**:
  - **Enhance Grounding**: Force the LLM to rely **strictly** on the provided documents (`context`) to generate an answer. This is the core principle for preventing fabricated information.
  - **Mitigate Hallucination**: Prevent the LLM from inventing information or using unverified external knowledge.
  - **Ensure Consistent Output Formatting**: Provide clear instructions on how to respond, especially when no relevant information is found.

---

#### 🧱 Prompt Structure

- **Code**:

  ```python
  prompt_template = """<role>
  You are an AI assistant specialized in extracting information from documents. Your task is to answer the question based STRICTLY on the provided content. Do not use any external knowledge.
  </role>

  <input>
      <documents>
      {context}
      </documents>

      <question>
      {question}
      </question>
  </input>

  <task>
  Carefully read the documents provided above. Based ONLY on the information within these documents, your task is to answer the user's question.
  </task>

  <instructions>
  1.  Your answer must be derived exclusively from the information present in the <documents> section. Do not infer or assume any information not explicitly stated.
  2.  If the documents do not contain information relevant to the question, you must answer exactly with the phrase: "I don't know". Do not attempt to guess or provide a partial answer.
  3.  Return ONLY the final answer itself. Do not add any introductory phrases, explanations, or conversational filler like "Based on the documents...".
  </instructions>
  """
  ```

---

#### 🧩 Detailed Component Analysis

- **Rationale for XML Tags (`<role>`, `<input>`, etc.)**:
  - Models like Claude and GPT-4 are trained to recognize this structure. Using XML tags helps the model clearly distinguish between different parts of the prompt: role, input data, and instructions. This significantly increases the model's ability to follow instructions.
- **`<role>`**:
  - **Purpose**: To set the "persona" and context for the LLM. This instruction transforms the LLM from a creative conversational assistant into a focused and serious information extraction specialist. Adding "Do not use any external knowledge" further reinforces this constraint.
- **`<input>`**:
  - **Purpose**: To encapsulate all input data, helping the LLM distinguish between background information (`documents`) and the request to be processed (`question`).
  - **`{context}`**: A placeholder for the text chunks retrieved from the vector database. This is the "Augmented" component in RAG.
  - **`{question}`**: A placeholder for the user's query.
- **`<task>`**:
  - **Purpose**: To issue a direct, concise command that re-emphasizes the primary mission. Repeating the task helps increase the LLM's focus on the objective.
- **`<instructions>`**:
  - **Purpose**: This is the most critical section for controlling the LLM's behavior.
  - **`1. ONLY use information...`**: The core directive for enforcing "grounding". Emphasizing with uppercase letters and words like "exclusively" adds weight to the command.
  - **`2. If no relevant information...`**: A crucial "guardrail." It provides a safe exit path, preventing the LLM from guessing when data is unavailable. For enterprise applications, an AI that reliably answers "I don't know" is far more trustworthy than one that gives a wrong answer.
  - **`3. Return ONLY the answer...`**: Ensures a clean, machine-parsable output, eliminating conversational filler.

---

#### ✅ Advantages and Limitations

- **Advantages**:
  - **High Reliability**: Very effective for factual question-answering (Q&A) tasks.
  - **Easy to Control**: The rigid structure helps minimize undesirable behaviors.
  - **Simplicity**: Straightforward to implement and debug.
- **Limitations**:
  - **Less Flexible**: Not suitable for tasks requiring complex reasoning, synthesis of information from multiple sources, or extended conversations.
  - **Dependent on Retrieval Quality**: If the retrieval stage fails and does not provide the correct context, this prompt has no self-correction mechanism.

---

</details>

---

## 🧠 Prompt Template for the ReAct Framework

---

<details>
<summary>Analysis of the structure and logic for an Agent using the ReAct framework</summary>

---

#### 🎯 Overview and Objectives

- **Strategy**: This prompt is not just for answering a question, but for empowering the LLM to become an **autonomous Agent**.
- **Operating Model**: It teaches the LLM to follow the **Thought -> Action -> Observation** cycle.
- **Primary Objectives**:
  - **Complex Problem Solving**: Allows the LLM to break down a large problem into smaller, sequential steps.
  - **Tool Use**: Grants the LLM the ability to proactively use external tools (e.g., `hybrid_retrieve`, `web_search`) to gather necessary information.
  - **Increase Transparency**: The "Thought" stream reveals the LLM's reasoning process, making it easier for humans to debug and understand its logic.

---

#### 🧱 Prompt Structure

- **Code**:

  ```python
  prompt_template = '''Answer the following questions as best you can. You have access to the following tools:

  {tools}

  Use the following format for your response. This format is mandatory.

  Question: the input question you must answer
  Thought: You should always think about what to do. Analyze the user's question and the previous steps. Formulate a plan to get closer to the final answer. If you need to use a tool, state which one and why.
  Action: The action to take, which must be one of [{tool_names}]
  Action Input: The input to the action. This should be a well-formed query or argument for the selected tool.
  Observation: The result of the action. This will be provided by the system.
  ... (this Thought/Action/Action Input/Observation cycle can repeat N times as you gather information)
  Thought: I have now gathered all necessary information and can provide a final answer.
  Final Answer: The final answer to the original input question. If no relevant information is available after thorough investigation, answer exactly: "I don't know". Return ONLY the answer — nothing else. If the question is a yes/no question, only return 'Yes' or 'No'.

  Begin!

  Question: {input}
  Thought:{agent_scratchpad}'''
  ```

---

#### 🧩 Detailed Component Analysis

- **`{tools}` and `{tool_names}`**:
  - **Purpose**: These are placeholders that a framework (e.g., LangChain) automatically populates.
  - **`{tools}`**: Contains a detailed description of each tool. **The quality of this description is extremely important.** A good description helps the LLM understand what the tool does and what its expected input is. For example, a description for "web_search" should state: "Searches the web for up-to-date information. Input should be a concise search query."
  - **`{tool_names}`**: Lists the names of the tools the LLM is allowed to choose in the `Action` step.
- **`Use the following format:`**:
  - **Purpose**: A "meta-instruction" that defines the entire iterative structure of the reasoning process. Emphasizing "This format is mandatory" enhances compliance.
- **`Thought:`**:
  - **Purpose**: Forces the LLM to "think out loud." This is where the LLM analyzes the situation, evaluates the information it has, and plans its next action. This is the **"Reasoning"** part of ReAct. A more detailed description in the prompt helps guide the LLM's thinking process.
- **`Action:` and `Action Input:`**:
  - **Purpose**: The LLM makes an executive decision: it selects a tool and provides the input for it. This is the **"Acting"** part of ReAct.
- **`Observation:`**:
  - **Purpose**: A placeholder for the result returned from the tool. This information will be used in the next `Thought` step to adjust the plan.
- **`... (this ... can repeat N times)`**:
  - **Purpose**: Explicitly indicates that this cycle can be repeated, enabling multi-hop reasoning.
- **`{agent_scratchpad}`**:
  - **Purpose**: A special variable that acts as the agent's "short-term memory." The framework automatically populates this with the history of previous `Thought/Action/Observation` cycles, helping the agent track its progress, learn from previous steps, and avoid repeating mistakes.

---

#### ✅ Strategic Comparison with Naive RAG

- **Information Flow**:
  - **Naive RAG**: A static, one-way flow. Context is provided only once.
    - `Retrieve -> Augment -> Generate`
  - **ReAct**: A dynamic and iterative flow. The agent actively seeks context as needed.
    - `(Thought -> Action -> Observation) -> (Thought -> ...) -> Final Answer`
- **Capability**:
  - **Naive RAG**: Best for **Question Answering** based on a fixed set of documents.
  - **ReAct**: Best for **Problem Solving** that requires lookups, synthesis, and complex reasoning.
- **Level of Control**:
  - **Naive RAG**: Provides tight control over the LLM's behavior.
  - **ReAct**: Grants the LLM higher autonomy, which is more powerful but can also lead to unexpected loops or inefficient actions if not designed carefully.

---

</details>
