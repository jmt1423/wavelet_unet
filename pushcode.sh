#!/bin/zsh

# get commit message
if [[ -z "${message//}" ]]
then
	message=$(date '+%Y-%m-%d %H:%M:%S')
fi


# stage all commits
git add .
echo 'all files staged for commit'

# commit changes
git commit -m "$message"
echo 'changed commited to repo'

# push to my remote repo
git push https://github.com/jmt1423/flower_image_segmentation.git
echo 'changes pushed to jon1 branch'
