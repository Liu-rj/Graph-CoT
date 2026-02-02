import os
from tqdm import tqdm
import logging
import argparse
import jsonlines
import datetime
import time
from datasets import load_dataset
from GraphAgent import GraphAgent
from tools.retriever import NODE_TEXT_KEYS
from graph_prompts import graph_agent_prompt, graph_agent_prompt_zeroshot

from IPython import embed

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str, default="dblp")
parser.add_argument("--openai_api_key", type=str, default="xxx")
parser.add_argument("--path", type=str)
parser.add_argument("--benchmark_dir", type=str)
parser.add_argument("--save_file", type=str)
parser.add_argument(
    "--embedder_name", type=str, default="sentence-transformers/all-mpnet-base-v2"
)
parser.add_argument("--faiss_gpu", type=bool, default=False)
parser.add_argument("--embed_cache", type=bool, default=True)
parser.add_argument("--max_steps", type=int, default=15)
parser.add_argument("--zero_shot", type=bool, default=False)
parser.add_argument("--ref_dataset", type=str, default=None)

parser.add_argument("--llm_version", type=str, default="gpt-3.5-turbo")
args = parser.parse_args()

args.embed_cache_dir = args.path
args.graph_dir = os.path.join(args.path, "graph_{id}.json")
# args.data_dir = os.path.join(args.path, "cypher_path_search.json")
# args.data_dir = os.path.join(args.path, "data_subset.json")
args.node_text_keys = NODE_TEXT_KEYS[args.dataset]
args.ref_dataset = args.dataset if not args.ref_dataset else args.ref_dataset

assert args.llm_version in [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-13b-chat-hf",
    "claude-3.5-sonnet",
    "llama-3.1-405b",
    "mistral-large",
    "deepseek-r1",
    "deepseek-chat",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]


def remove_fewshot(prompt: str) -> str:
    # prefix = prompt.split('Here are some examples:')[0]
    # suffix = prompt.split('(END OF EXAMPLES)')[1]
    prefix = prompt[-1].content.split("Here are some examples:")[0]
    suffix = prompt[-1].content.split("(END OF EXAMPLES)")[1]
    return prefix.strip("\n").strip() + "\n" + suffix.strip("\n").strip()


def main():
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    output_file_path = args.save_file
    parent_folder = os.path.dirname(output_file_path)
    parent_parent_folder = os.path.dirname(parent_folder)
    if not os.path.exists(parent_parent_folder):
        os.mkdir(parent_parent_folder)
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)

    agent_prompt = (
        graph_agent_prompt if not args.zero_shot else graph_agent_prompt_zeroshot
    )
    agent = GraphAgent(args, agent_prompt)

    dataset = load_dataset(f"rmanluo/RoG-{args.dataset}",
                           split="test",
                           cache_dir="datasets",  # Set your desired folder here
    )
    
    # resume from history
    start_id = 0
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            done_lines = f.readlines()
        done_count = len(done_lines)
        print(f"Resuming from {done_count} done questions.")
        dataset = dataset.select(range(done_count, len(dataset)))
        start_id = done_count

    for it, item in enumerate(dataset, start=start_id):
        print(f"Processing {it}-th question, id: {item['id']}")
        question = item["question"]
        id_mapping = {}
        for entity in item["q_entity"]:
            id_mapping[entity] = entity

        print(f"Question {it + 1}: {question}")
        agent.load_graph(args.graph_dir.format(id=it))
        agent.set_graph_funcs(agent.graph)

        tic = time.time()
        agent.run(question, id_mapping, item["a_entity"])
        duration = time.time() - tic
        print(f"Answer: {agent.answer}")
        print(f"Time taken: {duration}")
        print(f"Ground Truth Answer: {agent.key}")
        print("---------")

        result_entree = {
            "question_type": args.dataset,
            "question": question,
            "method": "GraphCoT",
            "model_answer": agent.answer,
            "duration": round(duration, 2),
            "token_count": agent.token_count,
            "api_calls": agent.api_calls,
            "gt_answer": item["a_entity"],
        }
        print(result_entree)

        with jsonlines.open(output_file_path, "a") as writer:
            writer.write(result_entree)


if __name__ == "__main__":
    main()
