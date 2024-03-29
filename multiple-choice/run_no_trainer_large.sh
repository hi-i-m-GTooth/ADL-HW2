# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

accelerate launch run_swag_no_trainer.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --train_file ../data/mc_train.json \
  --validation_file ../data/mc_valid.json \
  --output_dir ../mc_large \
  --learning_rate 3e-5 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --checkpointing_steps epoch \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --max_length 512 \
  --pad_to_max_length
