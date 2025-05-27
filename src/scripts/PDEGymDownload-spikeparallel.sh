#!/bin/bash


DATASETS=("NS-Gauss" "NS-Sines" "NS-PwC" "NS-SL" "NS-SVS" "NS-BB")
BASE_DIR="../../../data/fluidgpt"  

NUM_PARALLEL=6

for dataset in "${DATASETS[@]}"; do
    shortname="${dataset#NS-}"                 
    foldername="PDEGym-${shortname}"          
    target_dir="${BASE_DIR}/${foldername}"

    echo "Downloading dataset: $dataset → $foldername"
    mkdir -p "$target_dir"

    (
        cd "$target_dir"
        seq 0 16 | xargs -P "$NUM_PARALLEL" -I {} bash -c '
            file="velocity_{}.nc"
            url="https://huggingface.co/datasets/camlab-ethz/'"$dataset"'/resolve/main/$file"
            if [ -f "$file" ]; then
                echo "$file already exists in '"$foldername"', skipping."
            else
                echo "Downloading $file to '"$foldername"'..."
                curl -s -L -O "$url"
            fi
        '
    )
done