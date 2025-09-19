APIKEY=$1

data_path=[YOUR SYNTHETIC DATA PATH, NO .csv] # example: "../SOME_DIR/PE_iter_0", where the file is "PE_iter_0.csv"
python extract_metadata.py --load_path ${data_path}.csv --save_path ${data_path}_annotated.csv --API_KEY ${APIKEY}