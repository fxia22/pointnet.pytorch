SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/..
wget http://modelnet.cs.princeton.edu/ModelNet40.zip --no-check-certificate
unzip ModelNet40.zip
# rm ModelNet40.zip # in case you need to reuse
cd ..
cp test.txt trainval.txt ModelNet40/
