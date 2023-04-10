# agents.py
from collections import deque
import annoy

from typing import Dict, List
from config import OBJECTIVE
from openai_utils import openai_call, get_ada_embedding

def query_index(index, query_vector, n):
    nearest_ids, distances = index.get_nns_by_vector(query_vector, n, include_distances=True)
    results = []
    for i, item_id in enumerate(nearest_ids):
        results.append((item_id, distances[i]))
    return results





def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str]
):
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as an array."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]

def prioritization_agent(this_task_id: int, task_list):
    #global task_list  # Add this line to define task_list as a global variable
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    prompt = f"""
    You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split("\n")
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


def execution_agent(objective: str, task: str, index: annoy.AnnoyIndex, metadata: Dict):
    context = context_agent(query=objective, n=5, index=index, metadata=metadata)  # pass metadata as a keyword argument

    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}\n.
    Take into account these previously completed tasks: {context}\n.
    Your task: {task}\nResponse:"""
    return openai_call(prompt, temperature=0.7, max_tokens=2000)





def context_agent(query: str, n: int, index: annoy.AnnoyIndex, metadata: Dict):
    query_embedding = get_ada_embedding(query)
    results = query_index(index, query_embedding, n)
    sorted_results = sorted(results, key=lambda x: x[1])

    context = []
    for item in sorted_results:
        try:
            context.append(metadata[str(item[0])]["task"])
        except KeyError:
            print(f"Metadata for item {item[0]} not found.")
    return context


