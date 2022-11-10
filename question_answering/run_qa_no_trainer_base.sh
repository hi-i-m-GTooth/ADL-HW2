accelerate launch run_qa_no_trainer.py \
  --model_name_or_path bert-base-chinese \
  --train_file ../data/qa_train.json \
  --validation_file ../data/qa_valid.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 5 \
  --learning_rate 3e-5 \
  --checkpointing_steps epoch \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --output_dir ../qa_base \
  --my_report --seed 9876
  #--per_device_train_batch_size 1 \
  #--per_device_eval_batch_size 1 \