OPENAI_KEY=xxx
GPT_version=claude-3.5-sonnet
max_steps=20

DATASET=maple
SUBDATASET=Physics # Biology, Chemistry, Materials_Science, Medicine, Physics
DATA_PATH=/home/ubuntu/PolyG/examples/datasets/maple/$SUBDATASET
BENCHMARK=/home/ubuntu/PolyG/examples/benchmarks/physics
SAVE_FILE=/home/ubuntu/Graph-CoT/Graph-CoT/results/$GPT_version/maple-$SUBDATASET/results_rephrased.jsonl

CUDA_VISIBLE_DEVICES=0 python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --benchmark_dir $BENCHMARK \
                    --save_file $SAVE_FILE \
                    --llm_version $GPT_version \
                    --openai_api_key $OPENAI_KEY \
                    --max_steps $max_steps

DATASET=goodreads # legal, biomedical, amazon, goodreads, dblp
DATA_PATH=/home/ubuntu/PolyG/examples/datasets/$DATASET
BENCHMARK=/home/ubuntu/PolyG/examples/benchmarks/$DATASET
SAVE_FILE=/home/ubuntu/Graph-CoT/Graph-CoT/results/$GPT_version/$DATASET/results_rephrased.jsonl

CUDA_VISIBLE_DEVICES=0 python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --benchmark_dir $BENCHMARK \
                    --save_file $SAVE_FILE \
                    --llm_version $GPT_version \
                    --openai_api_key $OPENAI_KEY \
                    --max_steps $max_steps

DATASET=amazon # legal, biomedical, amazon, goodreads, dblp
DATA_PATH=/home/ubuntu/PolyG/examples/datasets/$DATASET
BENCHMARK=/home/ubuntu/PolyG/examples/benchmarks/$DATASET
SAVE_FILE=/home/ubuntu/Graph-CoT/Graph-CoT/results/$GPT_version/$DATASET/results_rephrased.jsonl

CUDA_VISIBLE_DEVICES=0 python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --benchmark_dir $BENCHMARK \
                    --save_file $SAVE_FILE \
                    --llm_version $GPT_version \
                    --openai_api_key $OPENAI_KEY \
                    --max_steps $max_steps
