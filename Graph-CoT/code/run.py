import os
from tqdm import tqdm
import logging
import argparse
import jsonlines
import datetime
import time
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
args.graph_dir = os.path.join(args.path, "graph.json")
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
    "claude-3-5-sonnet",
    "llama-3.1-405b",
]


def remove_fewshot(prompt: str) -> str:
    # prefix = prompt.split('Here are some examples:')[0]
    # suffix = prompt.split('(END OF EXAMPLES)')[1]
    prefix = prompt[-1].content.split("Here are some examples:")[0]
    suffix = prompt[-1].content.split("(END OF EXAMPLES)")[1]
    return prefix.strip("\n").strip() + "\n" + suffix.strip("\n").strip()


def main():
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    # with open(args.data_dir, "r") as f:
    #     contents = []
    #     for item in jsonlines.Reader(f):
    #         contents.append(item)

    ####################################################################
    # contents = [contents[80]]
    ####################################################################

    output_file_path = args.save_file

    parent_folder = os.path.dirname(output_file_path)
    parent_parent_folder = os.path.dirname(parent_folder)
    if not os.path.exists(parent_parent_folder):
        os.mkdir(parent_parent_folder)
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
    # if os.path.exists(output_file_path):
    #     os.remove(output_file_path)

    # if not os.path.exists("{}/logs".format(parent_folder)):
    #     os.makedirs("{}/logs".format(parent_folder))
    # logs_dir = "{}/logs".format(parent_folder)
    agent_prompt = (
        graph_agent_prompt if not args.zero_shot else graph_agent_prompt_zeroshot
    )
    agent = GraphAgent(args, agent_prompt)

    question_types = [
        # "single_entity_abstract_rephrased",
        # "single_entity_concrete_rephrased",
        # "multi_entity_abstract_rephrased",
        "multi_entity_concrete_rephrased",
        # "nested_question_rephrased"
    ]
    for question_type in question_types:
        print("############################################")
        print(f"Running for question type: {question_type}")
        contents = []
        with open(os.path.join(args.benchmark_dir, f"{question_type}.jsonl"), "r") as f:
            for item in jsonlines.Reader(f):
                contents.append(item)

        # unanswered_questions = []
        # correct_logs = []
        # halted_logs = []
        # incorrect_logs = []
        # generated_text = []
        for i in tqdm(range(len(contents))):
            print(f"Question {i + 1}: {contents[i]['question']}")
            tic = time.time()
            agent.run(
                contents[i]["question"], contents[i]["entity"], contents[i]["answer"]
            )
            duration = time.time() - tic
            print(f"Answer: {agent.answer}")
            print(f"Time taken: {duration}")
            print(f"Ground Truth Answer: {agent.key}")
            print("---------")
            # log = f"Question: {contents[i]['question']}\n"
            # log += (
            #     remove_fewshot(agent._build_agent_prompt())
            #     + f"\nCorrect answer: {agent.key}\n\n"
            #     if not args.zero_shot
            #     else agent._build_agent_prompt()[-1].content
            #     + f"\nCorrect answer: {agent.key}\n\n"
            # )
            # with open(os.path.join(logs_dir, contents[i]["qid"] + ".txt"), "w") as f:
            #     f.write(log)

            ## summarize the logs
            # if agent.is_correct():
            #     correct_logs.append(log)
            # elif agent.is_halted():
            #     halted_logs.append(log)
            # elif agent.is_finished() and not agent.is_correct():
            #     incorrect_logs.append(log)
            # else:
            #     raise ValueError("Something went wrong!")

            result_entree = {
                "question_type": question_type,
                "question": contents[i]["question"],
                "method": "GraphCoT",
                "model_answer": agent.answer,
                "duration": round(duration, 2),
                "token_count": agent.token_count,
                "api_calls": agent.api_calls,
                "gt_answer": contents[i]["answer"],
            }
            print(result_entree)

            # generated_text.append(result_entree)

            with jsonlines.open(output_file_path, "a") as writer:
                writer.write(result_entree)

        # print(
        #     f"Finished Trial {len(contents)}, Correct: {len(correct_logs)}, Incorrect: {len(incorrect_logs)}, Halted: {len(halted_logs)}"
        # )
        # print(
        #     "Unanswered questions {}: {}".format(
        #         len(unanswered_questions), unanswered_questions
        #     )
        # )


if __name__ == "__main__":
    main()
