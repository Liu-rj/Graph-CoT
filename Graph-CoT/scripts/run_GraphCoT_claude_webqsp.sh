OPENAI_KEY=xxx
GPT_version=claude-3.5-sonnet
max_steps=20

DATASET=webqsp
DATA_PATH=/home/ubuntu/PolyG/examples/datasets/webqsp
BENCHMARK=/home/ubuntu/PolyG/examples/benchmarks/physics
SAVE_FILE=/home/ubuntu/Graph-CoT/Graph-CoT/results/$GPT_version/$DATASET/results_rephrased.jsonl

CUDA_VISIBLE_DEVICES=0 python ../code/run_webqsp.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --benchmark_dir $BENCHMARK \
                    --save_file $SAVE_FILE \
                    --llm_version $GPT_version \
                    --openai_api_key $OPENAI_KEY \
                    --max_steps $max_steps
