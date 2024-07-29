cat instructions.cu | grep __device__ | awk '{ print $3 }' | awk -F"(" '{ print "map[\""$1"\"]=" NR-1";" }'
