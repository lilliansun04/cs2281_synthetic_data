while read oldrev
do
    echo "Processing commit $oldrev"
    git checkout $oldrev
    git rm --cached --ignore-unmatch synthetic/summary_train.csv synthetic/summary_clean.csv
    git commit --amend --no-edit
done < original_commits.txt
