echo "woah dude"
# setup android sdk
# ./tools/bin/sdkmanager --update
# ./tools/bin/sdkmanager 

# save ndk root path 
cd android-ndk-r13b
export NDK_ROOT=$PATH
cd ..

# install and save hexagon sdk root path
cd root/Qualcomm/Hexagon_SDK
export QUALCOMM_SDK=$PATH
cd ../../..

# save tf root path
cd tensorflow
export TF_ROOT_DIR=$PATH
cd ..

# run sample inception builder
chmod +x ./tensorflow/contrib/makefile/samples/build_and_run_inception_hexagon.sh
. ./tensorflow/contrib/makefile/samples/build_and_run_inception_hexagon.sh
