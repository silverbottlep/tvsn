#!/bin/bash
SHAPENET_DATA=$1
mkdir $1/new_chair

while IFS='' read -r line || [[ -n "$line" ]]; do
    mv $1/03001627/$line/ $1/new_chair/
done < "train_chair.txt"
while IFS='' read -r line || [[ -n "$line" ]]; do
    mv $1/03001627/$line/ $1/new_chair/
done < "test_chair.txt"
