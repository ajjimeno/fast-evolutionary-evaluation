cat instructions.cu | grep __device__ | awk '{ print $3 }' | awk -F"(" '{ print "pfuncs["NR-1"]="$1";" }'
