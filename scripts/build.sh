SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
echo $SCRIPTPATH

g++ -std=c++11 $SCRIPTPATH/../utils/render_balls_so.cpp -o $SCRIPTPATH/../utils/render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
