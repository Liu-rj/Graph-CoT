OPENAI_KEY=xxx
# GPT_version=claude-3.5-sonnet
# GPT_version=deepseek-chat
GPT_version=Qwen/Qwen3-14B
max_steps=20

# DATASET=maple
# SUBDATASET=physics # Biology, Chemistry, Materials_Science, Medicine, Physics
# DATA_PATH=/home/renjie/PolyG/datasets/$SUBDATASET
# BENCHMARK=/home/renjie/PolyG/benchmarks/$SUBDATASET
# SAVE_FILE=/home/renjie/Graph-CoT/Graph-CoT/results/$GPT_version/$SUBDATASET/results.jsonl

# CUDA_VISIBLE_DEVICES=0 python ../code/run.py --dataset $DATASET \
#                     --path $DATA_PATH \
#                     --benchmark_dir $BENCHMARK \
#                     --save_file $SAVE_FILE \
#                     --llm_version $GPT_version \
#                     --openai_api_key $OPENAI_KEY \
#                     --max_steps $max_steps

DATASET=goodreads # legal, biomedical, amazon, goodreads, dblp
DATA_PATH=/home/renjie/PolyG/datasets/$DATASET
BENCHMARK=/home/renjie/PolyG/benchmarks/$DATASET
SAVE_FILE=/home/renjie/Graph-CoT/Graph-CoT/results/$GPT_version/$DATASET/results.jsonl

CUDA_VISIBLE_DEVICES=0 python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --benchmark_dir $BENCHMARK \
                    --save_file $SAVE_FILE \
                    --llm_version $GPT_version \
                    --openai_api_key $OPENAI_KEY \
                    --max_steps $max_steps

DATASET=amazon # legal, biomedical, amazon, goodreads, dblp
DATA_PATH=/home/renjie/PolyG/datasets/$DATASET
BENCHMARK=/home/renjie/PolyG/benchmarks/$DATASET
SAVE_FILE=/home/renjie/Graph-CoT/Graph-CoT/results/$GPT_version/$DATASET/results.jsonl

CUDA_VISIBLE_DEVICES=0 python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --benchmark_dir $BENCHMARK \
                    --save_file $SAVE_FILE \
                    --llm_version $GPT_version \
                    --openai_api_key $OPENAI_KEY \
                    --max_steps $max_steps
