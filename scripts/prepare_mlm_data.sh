#!/bin/sh
echo "Combining comment_text field from both training files and test into one text-only file."
cd data || cd ../data || exit
DIRECTORY=mlm_text
if [ ! -d "$DIRECTORY" ]; then
  mkdir $DIRECTORY
fi
#csvtool namedcol comment_text jigsaw-toxic-comment-train.csv jigsaw-unintended-bias-train.csv|tail -n +2 > $DIRECTORY/jigsaw-joint_text.txt
csvtool namedcol comment_text jigsaw-toxic-comment-train.csv jigsaw-unintended-bias-train.csv | tail -n +2 >$DIRECTORY/jigsaw-combined-lm.txt
csvtool namedcol content test.csv | tail -n +2 >>$DIRECTORY/jigsaw-combined-lm.txt

echo "Converting validation file into one text-only file."
csvtool namedcol comment_text validation.csv | tail -n +2 >$DIRECTORY/jigsaw-validation-lm.txt
