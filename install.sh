DEST_DIR="/usr/include/odepack"

if [ "$(id -u)" -ne 0 ]; then
  echo "Please run as root using sudo"
  exit 1
fi

mkdir -p "$DEST_DIR"
cp include/odepack/*.hpp "$DEST_DIR"
echo "Installation complete. The header files are now located in $DEST_DIR"

cd ..
rm -rf "odepack"