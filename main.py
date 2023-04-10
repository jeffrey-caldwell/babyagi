import os
import time
from collections import deque
from typing import Dict

import annoy

from config import (
    OBJECTIVE,
    INITIAL_TASK,
    
    OPENAI_API_MODEL,
)

from openai_utils import openai_call, get_ada_embedding
from agents import task_creation_agent, prioritization_agent, execution_agent, context_agent, query_index
from annoy_utils import create_annoy_index, set_seed, add_item_to_annoy_index, build_annoy_index, query_annoy_index

# Configure Annoy
index_dimension = 1536
metric = "angular"
n_trees = 10
seed = 42

# Initialize Annoy index
index = annoy.AnnoyIndex(index_dimension, metric)

# Task list
task_list = deque([])

# Metadata and embeddings
metadata = {}
embeddings = {}

def add_task(task: Dict):
    task_list.append(task)

# Add your functions here (task_creation_agent, prioritization_agent, execution_agent, and context_agent)

# Add the first task
first_task = {"task_id": 1, "task_name": INITIAL_TASK}

add_task(first_task)

# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"], index, metadata)  # pass metadata as a keyword argument


        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in Annoy index
        enriched_result = {
            "data": result
        }  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = get_ada_embedding(
            enriched_result["data"]
        )  # get vector of the actual result extracted from the dictionary
        metadata[result_id] = {"task": task["task_name"]}
        add_item_to_annoy_index(index, this_task_id, vector)  # Store the result in the index



        # Step 3: Create new tasks and reprioritize task list
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_list],
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id, task_list)

        # Rebuild the entire index
        index = create_annoy_index(index_dimension, metric)
        set_seed(index, seed)
        for task in task_list:
            task_embedding = get_ada_embedding(task["task_name"])
            metadata[result_id] = {"task": task["task_name"]}
            add_item_to_annoy_index(index, this_task_id, vector)  # Store the result in the index

        build_annoy_index(index, n_trees)

    time.sleep(1)  # Sleep before checking the task list again



