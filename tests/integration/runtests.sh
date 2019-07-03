cd "$( dirname "${BASH_SOURCE[0]}" )"
echo "$(pwd)"
cd ..
cd ..
source venv/bin/activate || echo "activated"
mkdir -p tests/out
python bin/graphs.py --checkpoints tests/resources/kellis_list_exp.pickle --export tests/out/test.png
