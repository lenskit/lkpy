

if [ -r ml-100k/u.data ]; then
    echo "ML-100K already downloaded"
    exit 0
fi

wget --no-verbose -O ml-100k.zip http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
