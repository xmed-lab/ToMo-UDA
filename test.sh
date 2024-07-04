#!/bin/bash

path="checkpoints/"
output_file="result.txt"

# 清空或创建输出文件
> "$output_file"

# 遍历文件并运行测试
for weight_file in $path/res101pre_gpu7_heart*; do
    # 输出处理过程到文件
    echo "Processing $weight_file" >> "$output_file"
    # 执行测试并将结果输出到文件
    python test.py --path $weight_file | grep 'mAP:' >> "$output_file"
    # 分隔线
    echo "------------------------------------" >> "$output_file"
done

# 从output_file中提取每个文件的mAP值和文件名
declare -A file_map_scores
while read -r line; do
    if [[ $line == Processing* ]]; then
        current_file=$(echo $line | grep -oE "$path[^ ]+")
    fi
    if [[ $line == mAP:* ]]; then
        current_map=$(echo $line | grep -oE "[0-9.]+")
        file_map_scores["$current_file"]=$current_map
    fi
done < "$output_file"

# 寻找mAP值最大的文件
max_map=0
best_file=""
for file in "${!file_map_scores[@]}"; do
    if (( $(echo "${file_map_scores[$file]} > $max_map" | bc -l) )); then
        max_map=${file_map_scores[$file]}
        best_file=$file
    fi
done

echo "Best file: $best_file with mAP: $max_map"

# 删除mAP值不是最大的文件
for file in "${!file_map_scores[@]}"; do
    if [[ $file != $best_file ]]; then
        echo "Deleting $file with mAP: ${file_map_scores[$file]}"
        rm "$file"
    fi
done
