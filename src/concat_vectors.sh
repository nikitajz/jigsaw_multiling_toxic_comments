#!/bin/bash
# This script takes vector files with specified pattern and concatenates them into one
# It also handles header (token count and dimension)
token_count=0
res_file=wiki.align.comb.vec
for filename in ./wiki.*.align.vec; do
	echo "Processing file $filename"
	read -ra header_arr <<< $(head -n1 $filename)
	((token_count+= ${header_arr[0]}))
	dim=${header_arr[1]}
	echo "Token counts: ${header_arr[0]}"
	tail -n +2 $filename >> $res_file
done

echo "Adding header"
cat <(echo "$token_count $dim") $res_file | sponge $res_file
echo "Total tokens count: $token_count"
res_token_cnt="$(wc -l $res_file)"
echo "Resulting file line count $res_token_cnt" 
